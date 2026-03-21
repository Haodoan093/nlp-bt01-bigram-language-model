# BT02 - Danh gia hai bo POS tagger tren Brown corpus

## 1) Tai tap du lieu Brown corpus (NLTK)
Su dung NLTK de tai va doc Brown corpus voi bo nhan universal:

- Download resource: brown, universal_tagset
- Tap du lieu: `brown.tagged_sents(tagset="universal")`

## 2) Gan nhan tu loai bang 2 bo tagger
Theo yeu cau sua lai, Brown chi dung de test (khong chia train/test tu Brown).

Hai bo tagger da danh gia:

1. PerceptronTagger pretrained cua NLTK (gan nhan Penn Treebank, sau do map sang universal)
2. BigramTagger train tren Treebank + backoff UnigramTagger + DefaultTagger("NOUN")

Script thuc hien: bt2/bt02_pos_tagger_evaluation.py

## 3) Danh gia do chinh xac (precision, recall, macro-F1)
Ket qua chay thuc te (ngay 2026-03-14):

- So cau test (Brown): 57,340

### Tong hop ket qua

| Tagger | Precision (macro) | Recall (macro) | Macro-F1 | Accuracy |
|---|---:|---:|---:|---:|
| PerceptronTagger (pretrained) | 0.8241 | 0.8237 | 0.8205 | 0.9184 |
| BigramTagger (train Treebank) | 0.8003 | 0.7532 | 0.7688 | 0.8606 |

## Giai thich cac do do

- Precision (cho tung nhan): trong cac token duoc du doan la mot nhan X, ti le du doan dung la bao nhieu.
- Recall (cho tung nhan): trong cac token thuc su thuoc nhan X, mo hinh tim dung duoc bao nhieu.
- F1 (cho tung nhan): trung binh dieu hoa giua precision va recall.
- Macro-F1: lay trung binh F1 cua tat ca nhan, moi nhan duoc trong so nhu nhau (khong phu thuoc tan suat nhan).

Vi Brown corpus co nhieu nhan khong can bang tan suat, macro-F1 la do do hop ly de danh gia cong bang tren toan bo nhan POS.

## Nhan xet

- Brown duoc dung dung vai tro test set, khong dung de huan luyen.
- Trong thiet lap nay, PerceptronTagger pretrained cho ket qua tot hon BigramTagger train tu Treebank tren tat ca cac chi so.
