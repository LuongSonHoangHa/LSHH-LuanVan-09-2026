# LSHH-LuanVan-09-2026: Xây dựng mô hình đa mô thức thị giác - ngôn ngữ ghép và phát hiện sản phẩm trùng lặp từ ảnh và tiêu đề

> **Đề tài Luận văn tốt nghiệp**: Phát hiện và ghép nối các sản phẩm trùng lặp trong thương mại điện tử bằng phương pháp học đặc trưng đo lường đa mô thức (Multimodal Metric Learning) kết hợp mạng thị giác **NFNet-F0**, mô hình ngôn ngữ đa ngữ **multilingual-E5-large** và hàm mất mát khoảng cách góc **ArcFace**.

---

## 1. Giới thiệu tổng quan & Ý tưởng cốt lõi (General Idea)

### 1.1. Đặt vấn đề
Trong các sàn thương mại điện tử (Shopee, Tiki, Lazada, ...), cùng một sản phẩm vật lý thường được đăng tải bởi nhiều người bán khác nhau với:
- **Tiêu đề đa ngôn ngữ, không đồng nhất**: viết tắt, lỗi chính tả, thứ tự từ bị xáo trộn, xen lẫn tiếng Anh, tiếng Việt, tiếng Indonesia.
- **Hình ảnh đa dạng**: ảnh chụp studio, ảnh chụp thực tế, ảnh nền tùy chỉnh, chèn watermark, góc chụp và điều kiện chiếu sáng khác nhau.

**Mục tiêu**: Cho một sản phẩm sản phẩm gồm cặp `(image, title)`, hệ thống cần truy xuất và gom cụm tất cả các sản phẩm thuộc cùng một sản phẩm vật lý (`label_group`) trên toàn sàn.

### 1.2. Ý tưởng giải pháp kỹ thuật
1. **Kiến trúc hai nhánh độc lập (Two-Branch Late Fusion)**:
   - **Nhánh Thị giác (Vision)**: Sử dụng backbone **NFNet-F0** (không dùng BatchNorm, trích xuất đặc trưng sâu sắc) kết hợp lớp gom cụm thích nghi **GeM Pooling ($p=3$)** và tầng chiếu tuyến tính về không gian nhúng 512 chiều.
   - **Nhánh Ngôn ngữ (Language)**: Sử dụng mô hình biến đổi Transformer đa ngữ mạnh mẽ **multilingual-E5-large** (1024 chiều), đóng băng 18 tầng đầu và tinh chỉnh 6 tầng cuối cùng, chiếu về không gian 512 chiều.
2. **Học biểu diễn khoảng cách góc bằng ArcFace (Additive Angular Margin Loss)**:
   - Các vector nhúng được chuẩn hóa trên mặt cầu đơn vị siêu không gian ($L_2$-normalized).
   - Hàm ArcFace áp đặt biên độ góc $m = 0.50$ và tỷ lệ co giãn $s = 30.0$, buộc các sản phẩm cùng nhóm nhãn (`label_group`) co cụm lại cực kỳ chặt chẽ trong khi đẩy lùi góc phân tách giữa các nhóm khác nhau.
3. **Chiến lược hợp nhất hậu kỳ (Late Fusion) & Hậu xử lý truy vấn**:
   - Từng mô thức đưa ra các dự đoán độc lập với ngưỡng cosine tối ưu riêng biệt.
   - Ghép nối vector (Concatenation) và hợp tập kết quả (Union of Matches).
   - Khảo sát hậu xử lý **$\alpha$-QE (Alpha Query Expansion)** và **DBA (Database-side Augmentation)** để tinh chỉnh trọng tâm vector truy vấn.

---

## 2. Kiến trúc hệ thống (System Architecture)

```
                     ┌───────────────────────────────────────────┐
                     │           DỮ LIỆU ĐẦU VÀO                 │
                     │    [Hình ảnh sản phẩm]  +  [Tiêu đề]      │
                     └─────────────────────┬─────────────────────┘
                                           │
                 ┌─────────────────────────┴─────────────────────────┐
                 │                                                   │
                 ▼ (Ảnh: 224x224)                                    ▼ (Text: "query: " + Title)
    ┌───────────────────────────┐                       ┌───────────────────────────┐
    │     NFNet-F0 Backbone     │                       │   multilingual-E5-large   │
    │  - Frozen: stem + stage0,1│                       │  - Frozen: layers 0..17   │
    │  - Trainable: stage2,3    │                       │  - Trainable: layers 18..23│
    └────────────┬──────────────┘                       └────────────┬──────────────┘
                 │                                                   │
                 ▼                                                   ▼
    ┌───────────────────────────┐                       ┌───────────────────────────┐
    │     GeM Pooling (p=3)     │                       │       Mean Pooling        │
    └────────────┬──────────────┘                       └────────────┬──────────────┘
                 │                                                   │
                 ▼                                                   ▼
    ┌───────────────────────────┐                       ┌───────────────────────────┐
    │ Linear Projection (512-d) │                       │ Linear Projection (512-d) │
    │       BatchNorm1d         │                       │       BatchNorm1d         │
    └────────────┬──────────────┘                       └────────────┬──────────────┘
                 │                                                   │
                 ├─────────────────────────┬─────────────────────────┤
                 ▼                         ▼                         ▼
    ┌─────────────────────────┐           │           ┌─────────────────────────┐
    │  ArcFace Training Head  │           │           │  ArcFace Training Head  │
    │   (s = 30.0, m = 0.50)  │           │           │   (s = 30.0, m = 0.50)  │
    └─────────────────────────┘           │           └─────────────────────────┘
                                          ▼
                             ┌─────────────────────────┐
                             │    L2-Normalization     │
                             │ (Hypersphere Embedding) │
                             └────────────┬────────────┘
                                          │
                                          ▼
                             ┌─────────────────────────┐
                             │   Late Fusion & Cosine  │
                             │     Threshold Sweep     │
                             │   (Shopee Test & VN)    │
                             └─────────────────────────┘
```

---

## 3. Bộ dữ liệu & Chiến lược phân chia (Datasets & Split Strategy)

### 3.1. Dữ liệu sử dụng
- **Shopee quốc tế (`Shopee.csv`)**: 34.250 sản phẩm, 11.014 nhóm nhãn (`label_group`)
- **Shopee loại bỏ tiếng Indo (`shopee_no_indo.csv`)**: 25.459 sản phẩm, 8.307 nhóm nhãn (dữ liệu sau khi xử lý loại bỏ các nhóm có tiếng Indonesia).
- **Dữ liệu sản phẩm tiếng Việt (`vietnamese.csv`)**: 2.003 sản phẩm, 447 nhóm nhãn (dữ liệu thu thập thực tế trên Shopee Việt Nam).

### 3.2. Chiến lược chia dữ liệu (80/20 Group-Aware Split)
Nhằm tránh hiện tượng rò rỉ dữ liệu (data leakage) tuyệt đối giữa tập Train và Test:
1. **Tập Huấn luyện (80% Merged Train)**:
   - `merge_train.csv`: Kết hợp 80% Shopee + 80% Vietnamese (~28.396 dòng, ~9.011 nhóm).
   - `merge_train_no_indo.csv`: Kết hợp 80% Shopee (đã lọc Indo) + 80% Vietnamese.
2. **Tập Đánh giá Kiểm thử (20% Held-out Test - Đánh giá độc lập)**:
   - `shopee_test.csv` (20% Shopee gốc)
   - `shopee_no_indo_test.csv`: (20% tập Shopee đã lọc tiếng Indo).
   - `vietnamese_test.csv` (20% dữ liệu Việt Nam): Đánh giá mô hình trên dữ liệu tiếng Việt.

---

## 4. Cấu trúc thư mục mã nguồn

```
LSHH-LuanVan-09-2026/
├── data/                               # Dữ liệu bảng CSV (metadata & splits)
│   ├── raw/
│   │   ├── Shopee.csv                  # Dữ liệu gốc Shopee
│   │   ├── shopee_no_indo.csv          # Dữ liệu Shopee đã lọc bỏ tiếng Indo
│   │   └── vietnamese.csv              # Dữ liệu sản phẩm Việt Nam
│   └── processed/
│       ├── merge_train.csv             # Tập train hợp nhất (Shopee + VN)
│       ├── merge_train_no_indo.csv     # Tập train hợp nhất (No-Indo + VN)
│       ├── shopee_test.csv             # Tập test đánh giá Shopee
│       ├── shopee_no_indo_test.csv     # Tập test đánh giá Shopee No-Indo
│       └── vietnamese_test.csv         # Tập test đánh giá sản phẩm Việt Nam
│
├── dataPreprocess/                     # Các công cụ xử lý dữ liệu trước khi train
│  
├── src/                                # Mã nguồn chính của các mô hình
│   │
│   ├── ArcFace/                        # Tinh chỉnh ArcFace
│   │   ├── ArcFace_image_NFNet-f0/     # Mô hình thị giác NFNet-F0 + ArcFace
│   │   └── ArcFace_text_e5/            # Mô hình ngôn ngữ multilingual-E5-large + ArcFace
│   │       
│   ├── baselines/                      # Các mô hình gốc (Zero-shot)
│   │   └── E5/
│   │
│   └── multimodel/                     # Tích hợp đa mô thức
│       ├── CLIP/                       # kiến trúc CLIP
│       └── NFNetf0_E5/                 # kiến trúc NFNet-f0 + E5
│
├── tools/                              # Các công cụ hỗ trợ
│   └── image_phash.py                  # Tính toán perceptual hash của ảnh
│
├── Reference/                          # Tài liệu nghiên cứu & bài báo khoa học tham khảo
│
├── .gitignore                          # Cấu hình bỏ qua ảnh, weights, cache, file nén
└── README.md                           # Tài liệu tổng quan đề tài luận văn
```
