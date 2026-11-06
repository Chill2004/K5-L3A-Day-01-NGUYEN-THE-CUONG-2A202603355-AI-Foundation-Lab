# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV: 2A202603355**
- **Họ và tên: NGUYEN THE CUONG**
- **SEED** (in ở ô đầu notebook): 4350

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính

| Mô hình | F1 test | F1 SUB 40 | p95 (ms/tin) | USD/1.000 tin |
|---|---:|---:|---:|---:|
| BoW + Logistic Regression | 0.941 | 0.947 | 0.073 | 0 |
| BoW + MLP | 0.945 | 0.947 | 0.275 | 0 |
| MiniLM + kNN | 0.925 | 1.000 | 18.265 | 0 |
| MiniLM + Logistic Regression | 0.960 | 0.974 | 7.204 | 0 |
| MiniLM + MLP head | 0.961 | 1.000 | 7.384 | 0 |

## 3. Dự đoán và thực tế

| Thí nghiệm | Dự đoán của tôi | Kết quả thực tế | Vì sao lệch |
|---|---|---|---|
| Learning rate (`ml_lr`) | Tôi dự đoán `lr=0.001` sẽ làm loss giảm nhanh hơn và `lr=50` sẽ cho kết quả kém. | Sau 300 epoch, F1 test lần lượt là **0.000** và **0.941**. | Learning rate nhỏ tạo bước cập nhật nhỏ. Với cùng số epoch, mô hình ở `lr=0.001` chưa học đủ để nhận ra lớp spam ở ngưỡng 0.5, nên recall spam bằng 0. |
| Bỏ ReLU (`dl_relu`) | Tôi dự đoán F1 sẽ giảm khi bỏ ReLU. | F1 test của MLP có ReLU là **0.945**, còn khi bỏ ReLU là **0.950**. | Hai lớp Linear liên tiếp tạo thành một phép biến đổi affine. Với dữ liệu BoW của bài này, tính phi tuyến chưa cho thấy lợi ích về F1 trong lần chạy đã lưu. Mức chênh chỉ **0.005**. |
| Tin nhắn ngụy trang (`emb_tricky`) | Tôi dự đoán Encoder + Logistic Regression sẽ làm tốt hơn BoW. | Cả hai đúng **5/6 tin**, nhưng sai ở hai tin khác nhau. | Embedding nhận ra tin thúc giục xác nhận thông tin để giữ phần thưởng, nhưng bỏ sót tin mời nhận kỳ nghỉ miễn phí. BoW cho kết quả ngược lại. Khả năng biểu diễn ngữ nghĩa giúp ở một số mẫu, nhưng lợi thế còn phụ thuộc vào tin nhắn cụ thể. |

## 4. Phân tích 50 case khó

| Case trong bộ 50 tin | Nhãn thật → dự đoán | Cơ chế có thể khiến mô hình sai |
|---|---|---|
| #[ID 1] — [trích nội dung tin] | [nhãn thật] → [nhãn mô hình] | [Chỉ ra từ khóa hoặc ngữ cảnh cụ thể khiến mô hình nhầm.] |
| #[ID 2] — [trích nội dung tin] | [nhãn thật] → [nhãn mô hình] | [Chỉ ra dấu hiệu mô hình bỏ sót hoặc tín hiệu sai mà mô hình dựa vào.] |

## 5. Thử thách cá nhân

| Ngưỡng | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.2 | 0.889 | 0.923 | 0.906 |
| 0.5 | 1.000 | 0.769 | 0.870 |
| 0.8 | 1.000 | 0.625 | 0.769 |

Trong ba lựa chọn, tôi chọn 0.8 nếu ngân hàng ưu tiên giảm nguy cơ chặn nhầm OTP. Khi tăng ngưỡng trên cùng bộ xác suất, số tin hợp lệ bị gắn nhãn spam không thể tăng. Đánh đổi là mô hình có thể bỏ lọt thêm spam. Để biết ngưỡng này giảm chặn nhầm OTP đến mức nào, cần đo riêng số false positive trên nhóm tin OTP hợp lệ.

## 6. Phương án cho sản phẩm thật

MiniLM + Logistic Regression làm phương án để thử tải. Mười triệu SMS/ngày tương đương khoảng 116 tin/giây trung bình. Kết quả notebook đáp ứng mốc p95 < 50 ms/tin trong điều kiện benchmark của bài. Đổi lại cần chạy encoder và có độ trễ cao hơn.

## 7. AI log và điều đã học

| Phần | Tôi hỏi AI gì? | AI hỗ trợ gì? | Sửa / kiểm chứng |
|---|---|---|---|
| TODO 4.1–4.4 | Nhờ đọc notebook và bổ sung các TODO còn thiếu trong phần LLM. | Viết phần gọi API, tạo prompt zero-shot/few-shot, đọc nhãn và phân loại theo lô. | AI đã chạy hai ô kiểm tra với API giả lập. [Ghi thêm phần tôi tự kiểm tra trên Colab, nếu có.] |
| Báo cáo cuối | Hỏi `lab1_report.json` có đọc lịch sử sử dụng AI không. | Giải thích báo cáo lấy dữ liệu từ `_LOG` và `RESULTS` trong notebook. | Đối chiếu với lệnh `json.dump()` trong ô báo cáo cuối. |
| Thử thách đổi ngưỡng | Hỏi vì sao `model.predict_proba()` báo lỗi và nhờ viết cách so sánh ba ngưỡng. | Chỉ ra biến `model` đang là chuỗi. Viết đoạn dùng `predict_proba()` và `metrics()` của Phần 1. | [Điền kết quả tôi tự kiểm tra sau khi chạy lại ô thử thách.] |