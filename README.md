# LSHH-LuanVan-09-2026: Xây dựng mô hình đa mô thức thị giác - ngôn ngữ ghép và phát hiện sản phẩm trùng lặp từ ảnh và tiêu đề

> **Đề tài Luận văn tốt nghiệp**: Phát hiện và ghép nối các sản phẩm trùng lặp trong thương mại điện tử bằng phương pháp học đặc trưng đo lường đa mô thức (Multimodal Metric Learning) kết hợp mạng thị giác **NFNet-F0**, mô hình ngôn ngữ đa ngữ **multilingual-E5-large** và hàm mất mát khoảng cách góc **ArcFace**.

---

## 1. Giới thiệu tổng quan & Ý tưởng cốt lõi (General Idea)

### 1.1. Đặt vấn đề
Trong các sàn thương mại điện tử (Shopee, Tiki, Lazada, ...), cùng một sản phẩm vật lý thường được đăng tải bởi nhiều người bán khác nhau với:
- **Tiêu đề đa ngôn ngữ, không đồng nhất**: viết tắt, lỗi chính tả, thứ tự từ bị xáo trộn, xen lẫn tiếng Anh và tiếng Việt.
- **Hình ảnh đa dạng**: ảnh chụp studio, ảnh chụp thực tế, ảnh nền tùy chỉnh, chèn watermark, góc chụp và điều kiện chiếu sáng khác nhau.

**Mục tiêu**: Cho một sản phẩm sản phẩm gồm cặp `(image, title)`, hệ thống cần truy xuất và gom cụm tất cả các sản phẩm thuộc cùng một sản phẩm vật lý (`label_group`) trên toàn sàn.

### 1.2. Hai kiến trúc so sánh chính

Luận văn so sánh **hai hướng tiếp cận đa mô thức** đã được tinh chỉnh (fine-tuned) với ArcFace:

#### Kiến trúc A — Late Fusion (NFNet-F0 + multilingual-E5-large)
Hai encoder độc lập, mỗi mô thức xử lý riêng biệt rồi hợp nhất ở hậu kỳ:
- **Nhánh Thị giác**: **NFNet-F0** backbone → GeM Pooling (p=3) → Linear(512) → ArcFace
- **Nhánh Ngôn ngữ**: **multilingual-E5-large** (đóng băng 18/24 layers, finetune 6 layers cuối) → Mean Pooling → Linear(512) → ArcFace
- **Hợp nhất**: Late Fusion — cosine retrieval độc lập từng nhánh, ghép concat hoặc union kết quả

#### Kiến trúc B — CLIP (Joint Multimodal Encoder)
Một encoder duy nhất được huấn luyện sẵn để hiểu đồng thời ảnh và văn bản:
- **CLIP** (ViT-B/32) — joint image-text embedding space
- Tinh chỉnh với ArcFace trên tập huấn luyện (`merge_train_no_indo.csv`)
- Cosine retrieval trực tiếp trên embedding CLIP + ArcFace head

#### Hàm mất mát chung — ArcFace (Additive Angular Margin Loss)
Áp dụng cho cả hai kiến trúc:
$$L = -\log\frac{e^{s\cdot\cos(\theta_{y_i} + m)}}{e^{s\cdot\cos(\theta_{y_i} + m)} + \sum_{j \neq y_i} e^{s\cdot\cos\theta_j}}, \quad s=30.0,\ m=0.50$$

---

## 2. Kiến trúc hệ thống (System Architecture)

### Kiến trúc A — NFNet-F0 + E5 Late Fusion

```
     [Ảnh sản phẩm (224x224)]               [Tiêu đề — "query: " + text]
              │                                          │
              ▼                                          ▼
  ┌─────────────────────────┐              ┌──────────────────────────────┐
  │     NFNet-F0 Backbone   │              │   multilingual-E5-large      │
  │  Frozen: stem+stage0,1  │              │  Frozen: layers 0..17        │
  │  Trainable: stage2,3    │              │  Trainable: layers 18..23    │
  └────────────┬────────────┘              └────────────┬─────────────────┘
               │                                        │
               ▼                                        ▼
  ┌─────────────────────────┐              ┌──────────────────────────────┐
  │    GeM Pooling (p=3)    │              │        Mean Pooling          │
  │  → Linear(512) + BN     │              │   → Linear(512) + BN         │
  └────────────┬────────────┘              └────────────┬─────────────────┘
               │  [Train]  ArcFace Head                  │  [Train]  ArcFace Head
               │  [Infer]  L2-Normalize                  │  [Infer]  L2-Normalize
               │                                        │
               └─────────────────┬──────────────────────┘
                                 ▼
                  ┌──────────────────────────────┐
                  │   Late Fusion (Concat/Union)  │
                  │   Cosine Threshold Sweep      │
                  │   → Shopee Test & VN Test     │
                  └──────────────────────────────┘
```

### Kiến trúc B — CLIP Fine-tuned

```
  [Ảnh sản phẩm]    [Tiêu đề văn bản]
         │                  │
         ▼                  ▼
  ┌──────────────────────────────────┐
  │          CLIP Encoder            │
  │  ViT-B/32                        │
  │  Image Encoder + Text Encoder    │
  │  → Joint embedding space (512-d) │
  └──────────────────┬───────────────┘
                     │  [Train]  ArcFace Head (s=30, m=0.50)
                     │  [Infer]  L2-Normalize
                     ▼
       ┌──────────────────────────┐
       │   Cosine Threshold Sweep │
       │ → Shopee Test & VN Test  │
       └──────────────────────────┘
```

---

## 3. Kế hoạch đánh giá (Evaluation Plan)

| Phase | Mô hình | Kiểu | Tập dữ liệu train | Mục tiêu |
| :--- | :--- | :---: | :--- | :--- |
| **Phase 1A** | E5-large zero-shot | Text | _(không train)_ | Baseline text đa ngữ |
| **Phase 1B** | NFNet-F0 zero-shot | Image | _(không train)_ | Baseline image |
| **Phase 1C** | CLIP zero-shot | Multimodal | _(không train)_ | Baseline multimodal mạnh |
| **Phase 2A** | E5-large + ArcFace | Text fine-tuned | `merge_train_no_indo.csv` | Upper-bound text |
| **Phase 2B** | NFNet-F0 + ArcFace | Image fine-tuned | `merge_train_no_indo.csv` | Upper-bound image |
| **Phase 2C** | **NFNet-F0 + E5 + Late Fusion** | Multimodal (A) | `merge_train_no_indo.csv` | **So sánh chính** |
| **Phase 2D** | **CLIP + ArcFace** | Multimodal (B) | `merge_train_no_indo.csv` | **So sánh chính** |


## 4. Bộ dữ liệu & Chiến lược phân chia (Datasets & Split Strategy)

### 4.1. Dữ liệu sử dụng
- **Shopee loại bỏ tiếng Indo (`shopee_no_indo.csv`)**: 25.459 sản phẩm, 8.307 nhóm nhãn (`label_group`) — dữ liệu Shopee sau khi xử lý loại bỏ các nhóm có tiếng Indonesia, giữ lại sản phẩm tiếng Anh.
- **Dữ liệu sản phẩm tiếng Việt (`vietnamese.csv`)**: 2.003 sản phẩm, 447 nhóm nhãn (dữ liệu thu thập thực tế trên Shopee Việt Nam).

### 4.2. Chiến lược chia dữ liệu (80/20 Group-Aware Split)
Nhằm tránh hiện tượng rò rỉ dữ liệu (data leakage) tuyệt đối giữa tập Train và Test:
1. **Tập Huấn luyện (80% Merged Train)**:
   - `merge_train_no_indo.csv`: Kết hợp 80% Shopee (đã lọc Indo) + 80% Vietnamese.
2. **Tập Đánh giá Kiểm thử (20% Held-out Test - Đánh giá độc lập)**:
   - `shopee_no_indo_test.csv`: 20% tập Shopee đã lọc tiếng Indo.
   - `vietnamese_test.csv`: 20% dữ liệu Việt Nam (đánh giá mô hình trên dữ liệu tiếng Việt).

---

## 5. Cấu trúc thư mục mã nguồn

```
LSHH-LuanVan-09-2026/
├── data/                               # Dữ liệu bảng CSV (metadata & splits)
│   ├── raw/
│   │   ├── shopee_no_indo.csv          # Dữ liệu Shopee đã lọc bỏ tiếng Indo
│   │   └── vietnamese.csv              # Dữ liệu sản phẩm Việt Nam
│   └── processed/
│       ├── merge_train_no_indo.csv     # Tập train hợp nhất (No-Indo + VN)
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
