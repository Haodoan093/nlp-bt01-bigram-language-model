# BT01 – Mô hình ngôn ngữ Bigram âm tiết cho tiếng Việt

## Mô tả

Bài tập xây dựng **mô hình ngôn ngữ Bigram** ở mức **âm tiết** cho tiếng Việt, bao gồm:

1. Huấn luyện mô hình Bigram có **Laplace smoothing** từ corpus Wikipedia tiếng Việt
2. Tính xác suất câu `"Hôm nay trời đẹp lắm"`
3. Sinh câu tự động từ mô hình

---

## Tập dữ liệu (Corpus)

- **Nguồn:** [Wikipedia tiếng Việt](https://vi.wikipedia.org/) — tải tự động qua [Wikipedia REST API](https://vi.wikipedia.org/w/api.php)
- **Số bài tải:** 200 bài ngẫu nhiên (~122,000 ký tự)
- **Không cần tải file ngoài** — corpus được fetch tự động khi chạy notebook

---

## Yêu cầu

- Python 3.10+
- Các thư viện:

```
requests
tqdm
```

Cài đặt:

```bash
pip install requests tqdm
```

---

## Cách chạy

### Trong VS Code

Mở file `BT01_Bigram_Language_Model.ipynb` và nhấn **Run All** (cần kết nối Internet).

### Trong terminal

```bash
pip install jupyter requests tqdm
jupyter notebook BT01_Bigram_Language_Model.ipynb
```

> ⚠️ Cell tải Wikipedia mất khoảng **2–3 phút**.

### Dùng file input/output

- `input.txt`: chứa câu cần tính xác suất (một dòng).
- `output.txt`: notebook sẽ ghi kết quả xác suất/perplexity sau khi chạy cell ở Mục 4.

Bạn có thể thay nội dung `input.txt` bằng câu bất kỳ, rồi chạy lại notebook để cập nhật `output.txt`.

---

## Kiến trúc mô hình

| Thành phần | Chi tiết |
|---|---|
| Mức tokenise | Âm tiết (tách theo khoảng trắng) |
| Ký hiệu đặc biệt | `<s>` (đầu câu), `</s>` (cuối câu) |
| Mô hình | Bigram: P(wᵢ \| wᵢ₋₁) |
| Smoothing | Laplace (add-1): P(w₂\|w₁) = (C(w₁,w₂)+1) / (C(w₁)+V) |

---

## Kết quả

### Xác suất câu "Hôm nay trời đẹp lắm"

| Bigram | P(wᵢ \| wᵢ₋₁) |
|---|---|
| P(hôm \| `<s>`) | 0.00018779 |
| P(nay \| hôm) | 0.00029394 |
| P(trời \| nay) | 0.00029351 |
| P(đẹp \| trời) | 0.00029369 |
| P(lắm \| đẹp) | 0.00029386 |
| P(`</s>` \| lắm) | 0.00029403 |
| **P(câu)** | **4.111 × 10⁻²²** |
| **log P(câu)** | **-49.2431** |
| **Perplexity** | **3667.21** |

> Xác suất thấp do corpus thiên về bài viết khoa học/địa lý trên Wikipedia; các từ "hôm nay trời đẹp lắm" xuất hiện rất ít. Laplace smoothing đảm bảo không bị zero probability.

### Xác suất các câu khác

| Câu | log P(s) | Perplexity |
|---|---|---|
| Hôm nay trời đẹp lắm | -49.24 | 3667.21 |
| Việt Nam là một đất nước xinh đẹp | -64.57 | 1304.92 |
| Tôi đang học xử lý ngôn ngữ tự nhiên | -77.57 | 2337.87 |
| Hà Nội là thủ đô của Việt Nam | -68.10 | 1933.18 |
| Mô hình ngôn ngữ rất hữu ích | -60.46 | 1915.04 |

### Một số câu sinh ra từ mô hình

**Sinh ngẫu nhiên (temperature = 1.0):**
1. `thành phần mềm đưa điền triều trở thành hai bên 45 km về phía đông`
2. `siêu cúp bóng đá độ 39 18 tháng 5 xã này được đánh chìm trong`
3. `sargramostim được westerman miêu tả khoa học harvard australian plant names index`

**Sinh có từ hạt giống:**
- seed=`hôm` → `hôm đó tiến đến việc này bạn đặt câu lạc tại wikispecies`
- seed=`việt` → `việt nam việt nam và inward link được xem thêm`
- seed=`hà` → `hà đông và được peshkova mô tả khoa học đầu nhiệm kỳ`

---

## Thông tin mô hình sau khi huấn luyện

| Thông số | Giá trị |
|---|---|
| Số câu huấn luyện | 1,924 |
| Kích thước từ vựng V | 3,401 âm tiết |
| Tổng token | 26,711 |
| Số bigram khác nhau | 13,218 |
