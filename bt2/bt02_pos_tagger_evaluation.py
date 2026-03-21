from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import mean
from typing import Iterable

import nltk
from nltk import pos_tag_sents
from nltk.corpus import brown, treebank
from nltk.tag import BigramTagger, DefaultTagger, UnigramTagger
from nltk.tag.mapping import map_tag


LOCAL_NLTK_DIR = Path(__file__).resolve().parent / "nltk_data"
OUTPUT_PATH = Path(__file__).resolve().parent / "output.txt"


@dataclass
class Metrics:
    precision: float
    recall: float
    f1: float


def ensure_nltk_resources() -> None:
    LOCAL_NLTK_DIR.mkdir(parents=True, exist_ok=True)
    local_data_path = str(LOCAL_NLTK_DIR)

    if local_data_path not in nltk.data.path:
        nltk.data.path.insert(0, local_data_path)

    nltk.download("brown", download_dir=local_data_path, quiet=True)
    nltk.download("treebank", download_dir=local_data_path, quiet=True)
    nltk.download("universal_tagset", download_dir=local_data_path, quiet=True)
    nltk.download("averaged_perceptron_tagger", download_dir=local_data_path, quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", download_dir=local_data_path, quiet=True)


def to_universal_from_penn(tag: str) -> str:
    return map_tag("en-ptb", "universal", tag)


def flatten_tags(
    gold_sents: Iterable[list[tuple[str, str]]],
    pred_sents: Iterable[list[tuple[str, str]]],
) -> tuple[list[str], list[str]]:
    y_true: list[str] = []
    y_pred: list[str] = []

    for gold_sent, pred_sent in zip(gold_sents, pred_sents):
        if len(gold_sent) != len(pred_sent):
            raise ValueError("Gold and predicted sentence lengths do not match.")
        for (_, gold_tag), (_, pred_tag) in zip(gold_sent, pred_sent):
            y_true.append(gold_tag)
            y_pred.append(pred_tag)

    return y_true, y_pred


def precision_recall_f1_per_label(
    y_true: list[str], y_pred: list[str], labels: list[str]
) -> dict[str, Metrics]:
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for true_tag, pred_tag in zip(y_true, y_pred):
        if true_tag == pred_tag:
            tp[true_tag] += 1
        else:
            fp[pred_tag] += 1
            fn[true_tag] += 1

    results: dict[str, Metrics] = {}
    for label in labels:
        p = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0.0
        r = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        results[label] = Metrics(precision=p, recall=r, f1=f1)

    return results


def macro_average(per_label: dict[str, Metrics]) -> Metrics:
    return Metrics(
        precision=mean(m.precision for m in per_label.values()),
        recall=mean(m.recall for m in per_label.values()),
        f1=mean(m.f1 for m in per_label.values()),
    )


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def format_metrics(name: str, macro: Metrics, acc: float) -> str:
    return (
        f"{name:<18} | Precision: {macro.precision:.4f} | "
        f"Recall: {macro.recall:.4f} | Macro-F1: {macro.f1:.4f} | Accuracy: {acc:.4f}"
    )


def evaluate(
    name: str,
    gold_sents: list[list[tuple[str, str]]],
    pred_sents: list[list[tuple[str, str]]],
    labels: list[str],
) -> tuple[Metrics, float]:
    y_true, y_pred = flatten_tags(gold_sents, pred_sents)
    per_label = precision_recall_f1_per_label(y_true, y_pred, labels)
    macro = macro_average(per_label)
    acc = accuracy(y_true, y_pred)
    print(format_metrics(name, macro, acc))
    return macro, acc


def write_output(
    test_sentences: int,
    perceptron_macro: Metrics,
    perceptron_acc: float,
    bigram_macro: Metrics,
    bigram_acc: float,
) -> None:
    content = (
        "BT02 - Evaluation of Two POS Taggers on the Brown Corpus\n"
        f"Run date: {date.today().isoformat()}\n\n"
        "Dataset:\n"
        "- Corpus: Brown (NLTK)\n"
        "- Tagset: universal\n"
        "- Evaluation setup: Brown is used only as the test set\n"
        f"- Number of test sentences: {test_sentences:,}\n\n"
        "Results:\n"
        "1) PerceptronTagger (pretrained)\n"
        f"   - Precision (macro): {perceptron_macro.precision:.4f}\n"
        f"   - Recall (macro):    {perceptron_macro.recall:.4f}\n"
        f"   - Macro-F1:          {perceptron_macro.f1:.4f}\n"
        f"   - Accuracy:          {perceptron_acc:.4f}\n\n"
        "2) BigramTagger (trained on Treebank)\n"
        f"   - Precision (macro): {bigram_macro.precision:.4f}\n"
        f"   - Recall (macro):    {bigram_macro.recall:.4f}\n"
        f"   - Macro-F1:          {bigram_macro.f1:.4f}\n"
        f"   - Accuracy:          {bigram_acc:.4f}\n\n"
        "Conclusion:\n"
        "- Brown was not used for training, only for testing.\n"
        "- In this setup, the pretrained PerceptronTagger outperforms BigramTagger trained on Treebank.\n"
    )
    OUTPUT_PATH.write_text(content, encoding="utf-8")


def main() -> None:
    ensure_nltk_resources()

    # Brown is used only as a test set.
    gold_test_sents = brown.tagged_sents(tagset="universal")
    test_untagged_sents = brown.sents()

    labels = sorted({tag for sent in gold_test_sents for _, tag in sent})

    perceptron_pred_ptb = pos_tag_sents(test_untagged_sents, lang="eng")
    perceptron_pred_universal = [
        [(word, to_universal_from_penn(tag)) for word, tag in sent]
        for sent in perceptron_pred_ptb
    ]

    treebank_train_sents = treebank.tagged_sents(tagset="universal")
    default_tagger = DefaultTagger("NOUN")
    unigram_backoff = UnigramTagger(treebank_train_sents, backoff=default_tagger)
    bigram_tagger = BigramTagger(treebank_train_sents, backoff=unigram_backoff)
    bigram_pred_universal = bigram_tagger.tag_sents(test_untagged_sents)

    print("Brown Corpus POS Tagging Evaluation (Brown used only for test)")
    print(f"Test sentences: {len(gold_test_sents):,}")
    print("-" * 110)
    perceptron_macro, perceptron_acc = evaluate(
        "PerceptronTagger", gold_test_sents, perceptron_pred_universal, labels
    )
    bigram_macro, bigram_acc = evaluate(
        "BigramTagger(Treebank)", gold_test_sents, bigram_pred_universal, labels
    )

    write_output(
        test_sentences=len(gold_test_sents),
        perceptron_macro=perceptron_macro,
        perceptron_acc=perceptron_acc,
        bigram_macro=bigram_macro,
        bigram_acc=bigram_acc,
    )


if __name__ == "__main__":
    main()
