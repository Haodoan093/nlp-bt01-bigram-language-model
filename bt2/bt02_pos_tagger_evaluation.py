from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

import nltk
from nltk.corpus import brown
from nltk.tag import BigramTagger, DefaultTagger, UnigramTagger


LOCAL_NLTK_DIR = Path(__file__).resolve().parent / "nltk_data"


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
    nltk.download("universal_tagset", download_dir=local_data_path, quiet=True)


def split_train_test(tagged_sents: list[list[tuple[str, str]]], train_ratio: float = 0.8):
    split_idx = int(len(tagged_sents) * train_ratio)
    return tagged_sents[:split_idx], tagged_sents[split_idx:]


def flatten_tags(
    gold_sents: Iterable[list[tuple[str, str]]],
    pred_sents: Iterable[list[tuple[str, str]]],
) -> tuple[list[str], list[str]]:
    y_true: list[str] = []
    y_pred: list[str] = []

    for gold_sent, pred_sent in zip(gold_sents, pred_sents):
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


def main() -> None:
    ensure_nltk_resources()

    tagged_sents = brown.tagged_sents(tagset="universal")
    train_sents, test_sents = split_train_test(tagged_sents, train_ratio=0.8)

    default_tagger = DefaultTagger("NOUN")

    unigram_tagger = UnigramTagger(train_sents, backoff=default_tagger)

    bigram_backoff = UnigramTagger(train_sents, backoff=default_tagger)
    bigram_tagger = BigramTagger(train_sents, backoff=bigram_backoff)

    test_untagged = brown.sents()[len(train_sents) :]

    unigram_pred = unigram_tagger.tag_sents(test_untagged)
    bigram_pred = bigram_tagger.tag_sents(test_untagged)

    labels = sorted({tag for sent in test_sents for _, tag in sent})

    uni_true, uni_pred = flatten_tags(test_sents, unigram_pred)
    bi_true, bi_pred = flatten_tags(test_sents, bigram_pred)

    uni_per_label = precision_recall_f1_per_label(uni_true, uni_pred, labels)
    bi_per_label = precision_recall_f1_per_label(bi_true, bi_pred, labels)

    uni_macro = macro_average(uni_per_label)
    bi_macro = macro_average(bi_per_label)

    uni_acc = accuracy(uni_true, uni_pred)
    bi_acc = accuracy(bi_true, bi_pred)

    print("Brown Corpus POS Tagging Evaluation (Universal tagset)")
    print(f"Train sentences: {len(train_sents):,} | Test sentences: {len(test_sents):,}")
    print("-" * 110)
    print(format_metrics("UnigramTagger", uni_macro, uni_acc))
    print(format_metrics("BigramTagger", bi_macro, bi_acc))


if __name__ == "__main__":
    main()
