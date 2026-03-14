# BT02 - Danh gia hai bo POS tagger tren Brown corpus

## 1) Tai tap du lieu Brown corpus (NLTK)
Su dung NLTK de tai va doc Brown corpus voi bo nhan universal:

- Download resource: brown, universal_tagset
- Tap du lieu: `brown.tagged_sents(tagset="universal")`

## 2) Gan nhan tu loai bang 2 bo tagger
Thuc hien tren Brown corpus voi chia tap:

- Train: 80% cau dau
- Test: 20% cau cuoi

Hai bo tagger da danh gia:

1. UnigramTagger + backoff DefaultTagger("NOUN")
2. BigramTagger + backoff UnigramTagger + DefaultTagger("NOUN")

Script thuc hien: bt2/bt02_pos_tagger_evaluation.py

## 3) Danh gia do chinh xac (precision, recall, macro-F1)
Ket qua chay thuc te (ngay 2026-03-14):

- So cau train: 45,872
- So cau test: 11,468

### Tong hop ket qua

| Tagger | Precision (macro) | Recall (macro) | Macro-F1 | Accuracy |
|---|---:|---:|---:|---:|
| UnigramTagger | 0.8973 | 0.8646 | 0.8660 | 0.9366 |
| BigramTagger | 0.9018 | 0.8686 | 0.8693 | 0.9455 |

## Giai thich cac do do

- Precision (cho tung nhan): trong cac token duoc du doan la mot nhan X, ti le du doan dung la bao nhieu.
- Recall (cho tung nhan): trong cac token thuc su thuoc nhan X, mo hinh tim dung duoc bao nhieu.
- F1 (cho tung nhan): trung binh dieu hoa giua precision va recall.
- Macro-F1: lay trung binh F1 cua tat ca nhan, moi nhan duoc trong so nhu nhau (khong phu thuoc tan suat nhan).

Vi Brown corpus co nhieu nhan khong can bang tan suat, macro-F1 la do do hop ly de danh gia cong bang tren toan bo nhan POS.

## Nhan xet

- BigramTagger tot hon UnigramTagger tren ca 3 do do macro va accuracy.
- Ly do: BigramTagger khai thac them ngu canh tu truoc do, nen giam nham lan trong cac truong hop mot tu co nhieu kha nang tu loai.
- Chenh lech khong qua lon vi backoff Unigram da manh, nhung Bigram van tao cai thien on dinh.
