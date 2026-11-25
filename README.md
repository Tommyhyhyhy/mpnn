
**Đề tài:** Phát hiện bài báo bị ghi sai tác giả bằng NLP
**Môn:** Mô phỏng ngẫu nhiên


---

## 1. Tóm tắt trong 30 giây

> Có một bài báo ghi tên tác giả A. Mình muốn biết A có thật sự viết bài đó không.
>
> Cách làm: so **thói quen dùng từ** của bài báo với thói quen của A. Nếu khác nhiều hơn mức "bình thường" thì nghi ngờ ghi sai tác giả.
>
> Cái khó là không ai biết "bình thường" là bao nhiêu. Vì vậy mình **mô phỏng** nó: lấy chính các bài thật của A, giả làm bài mới, lặp lại 1.000 lần để xem khoảng cách dao động thế nào.
>
> Kết quả: phát hiện được **84%** bài bị ghi sai.

---

## 2. Bài toán, bằng ví dụ đời thường

Hãy tưởng tượng bạn là biên tập viên một tờ báo có 50 phóng viên. Một ngày bạn nghi một bài đứng tên anh A thực ra do người khác viết. Có thể do biên tập nhầm tên, do có người viết hộ, hoặc do đạo văn.

Bạn không thể đọc tay hàng nghìn bài để kiểm tra. Bạn cần một cách **tự động** và **có căn cứ thống kê** để nói: *"Bài này đáng ngờ, p-value = 0,001."*

Đó chính là đồ án này.

---

## 3. Ý tưởng 1: "Dấu vân tay văn phong" (phần NLP)

Mỗi người có thói quen nói riêng. Có người hay nói "thì là mà", có người hay nói "kiểu như là". Viết cũng vậy.

Điều thú vị là thói quen này lộ rõ nhất ở **những từ nhỏ nhất**, gọi là hư từ: *the, of, and, to, in, a…*

- Nội dung bài (kinh tế, chính trị…) thay đổi theo đề tài.
- Nhưng tỷ lệ dùng *the*, *of*, *and* của một người thì khá ổn định, vì người ta dùng chúng một cách vô thức.

**Đồ án làm như sau:**

1. Lấy **300 từ xuất hiện nhiều nhất** trong toàn bộ kho bài. Phần lớn là hư từ.
2. Với mỗi bài báo, đếm xem mỗi từ chiếm bao nhiêu phần trăm. Ví dụ: *the* chiếm 6,1%, *of* chiếm 2,8%…
3. Như vậy **mỗi bài báo biến thành một dãy 300 con số**. Đây là "dấu vân tay" của bài.
4. "Hồ sơ" của tác giả A là **trung bình** dấu vân tay của các bài A đã viết.
5. Đo **khoảng cách** giữa dấu vân tay của bài cần kiểm tra và hồ sơ của A. Đồ án dùng thước đo tên là **Cosine Delta**:
   - khoảng cách nhỏ thì văn phong giống A,
   - khoảng cách lớn thì văn phong khác A.

> **Cosine Delta là gì?** Coi hai dãy 300 số là hai mũi tên trong không gian. Cosine Delta đo góc lệch giữa hai mũi tên. Cùng hướng thì bằng 0, càng lệch thì càng lớn. Đây là bản cải tiến của Burrows' Delta, thước đo kinh điển trong ngành phân tích văn phong.

NLP trong đồ án chỉ có vậy: **biến chữ thành số**. Phần chính là phần tiếp theo.

---

## 4. Ý tưởng 2: Phải biết "bình thường" là bao nhiêu (kiểm định giả thuyết)

Giả sử bài đang xét có khoảng cách tới A là **0,83**. Con số này là lớn hay nhỏ?

Không thể trả lời nếu chưa biết **các bài thật của A thường cách hồ sơ của A bao xa**. Chính A cũng không viết bài nào giống hệt bài nào.

Giống như đo nhiệt độ cơ thể: 37,5°C có phải sốt không? Phải biết thân nhiệt bình thường dao động trong khoảng nào thì mới kết luận được.

Trong thống kê, việc này gọi là **kiểm định giả thuyết**:

| | Nghĩa |
|---|---|
| **H0** (giả thuyết gốc) | Bài này đúng là của A |
| **H1** (đối thuyết) | Bài này bị ghi sai tác giả |
| **Thống kê T** | Khoảng cách văn phong giữa bài và hồ sơ của A |
| **Phân phối null** | Các giá trị T sẽ có **nếu H0 đúng**, tức "mức dao động bình thường" |
| **p-value** | Xác suất gặp khoảng cách lớn như vậy (hoặc lớn hơn) **nếu bài đúng là của A** |

**Quy tắc:** nếu p-value ≤ 0,05 thì bác bỏ H0 và kết luận bài đáng ngờ.

> **Hiểu p-value bằng câu nói:** p = 0,001 nghĩa là "nếu bài này thật sự của A, chỉ có 1 phần nghìn khả năng văn phong lệch nhiều đến vậy". Khả năng quá nhỏ nên ta nghi ngờ.

---

## 5. Ý tưởng 3: Mô phỏng "mức bình thường" bằng bootstrap 

**Vấn đề:** phân phối null **không có công thức toán**. Nó phụ thuộc vào từng tác giả, độ dài bài và nhiều thứ khác. Không thể tra bảng như phân phối chuẩn.

**Giải pháp: tự mô phỏng nó.** Với tác giả A, lặp lại 1.000 lần các bước sau:

```
Lặp 1.000 lần:
  1. Bốc ngẫu nhiên 1 bài THẬT của A, giả vờ đó là "bài mới cần kiểm tra"
  2. Lấy các bài còn lại của A, rút ngẫu nhiên CÓ HOÀN LẠI (bootstrap)
     để dựng một "hồ sơ A" hơi khác mỗi lần
  3. Cắt bài ở bước 1 cho dài đúng bằng bài cần kiểm tra
  4. Tính khoảng cách T* giữa bài đó và hồ sơ đó
→ Được 1.000 giá trị T*: đây chính là "mức dao động bình thường" của A
```

Vì bài ở bước 1 **đúng là của A**, 1.000 giá trị T* cho biết khoảng cách thường gặp khi H0 đúng.

**Tính p-value:**

```
p = (1 + số lần T* ≥ T của bài đang xét) / (1.000 + 1)
```

Ví dụ: bài đang xét có T = 1,05, và trong 1.000 lần mô phỏng không lần nào T* vượt 1,05.
Khi đó p = (1 + 0) / 1.001 ≈ 0,001, nên bác bỏ H0.

> **Vì sao gọi là bootstrap?** Tên này lấy từ câu "tự nhấc mình lên bằng dây giày". Ta không có thêm dữ liệu, nên tự lấy mẫu lại từ chính dữ liệu đang có để mô phỏng sự ngẫu nhiên. Phương pháp do Efron đề xuất năm 1979.

> **Vì sao phải cắt bài cho dài bằng bài cần kiểm tra (bước 3)?** Bài ngắn thì ít từ, tần suất từ dao động mạnh hơn, nên khoảng cách tự nhiên lớn hơn. Nếu không khớp độ dài, bài ngắn sẽ bị báo nhầm hàng loạt. Thí nghiệm TN2 chứng minh điều này.

---

## 6. Hai tầng mô phỏng (hãy nói rõ điều này khi bảo vệ)

| Tầng | Mô phỏng để làm gì | Ở đâu |
|---|---|---|
| **Tầng 1** | Tạo phân phối null bằng bootstrap để **tính p-value** | Bên trong phương pháp (mục 5) |
| **Tầng 2** | Tạo hàng nghìn tình huống **biết trước đáp án** để **chấm điểm chính phương pháp** | 9 thí nghiệm (mục 7) |

**Vì sao cần tầng 2?** Dữ liệu thật không có bài nào được đánh dấu là "bị ghi sai". Vì vậy mình **tự tạo** bài ghi sai: lấy bài của B rồi cố ý ghi tên A. Nhờ biết trước đáp án, mình đo được phương pháp đúng hay sai bao nhiêu phần trăm.

---

## 7. Chín thí nghiệm: câu hỏi, cách làm, kết quả

Thuật ngữ cần dùng:

- **Power** (năng lực phát hiện) = tỷ lệ bài **ghi sai** bị phát hiện. Càng cao càng tốt.
- **Sai lầm loại I** (báo động nhầm) = tỷ lệ bài **ghi đúng** nhưng bị nghi oan. Nên ≤ 5%.

### TN1: Phương pháp có hoạt động không?
- **Cách làm:** 2.500 bài kiểm tra, mỗi bài thử một lần ghi đúng tên và một lần ghi sai tên (tác giả sai chọn ngẫu nhiên).
- **Kết quả:** phát hiện **83,9%** bài ghi sai. AUC = 0,94 (1 là hoàn hảo, 0,5 là đoán mò). Thước đo cũ (Delta cổ điển) chỉ phát hiện được 29,9%.
- **Nhưng:** báo động nhầm là **7,7%**, cao hơn mức 5% mong muốn. TN1b giải thích lý do.

### TN1b: Vì sao báo động nhầm vượt 5%?
- **Cách làm:** so hai loại bài: bài **cùng giai đoạn** với bài mẫu, và bài **khác giai đoạn**.
- **Kết quả:** cùng giai đoạn chỉ báo nhầm **3,5%** (đạt yêu cầu). Khác giai đoạn thì lên **6,2%**.
- **Ý nghĩa:** thuật toán không sai. Văn phong và đề tài của nhà báo **thay đổi theo thời gian** nên bài mới hơi lệch khỏi hồ sơ cũ.

### TN2: Bài cần dài bao nhiêu chữ?
- **Cách làm:** cắt bài còn 50, 100, 200, 300, 400 từ.
- **Kết quả:** cần khoảng **300 từ** để phát hiện được 80%. Nếu **không khớp độ dài** khi mô phỏng, bài 50 từ bị báo nhầm tới **48%**, tức cứ 2 bài đúng thì 1 bài bị nghi oan.
- **Ý nghĩa:** đây là bằng chứng rõ nhất rằng **mô phỏng phải đúng điều kiện**.

### TN3: Tác giả cần bao nhiêu bài mẫu?
- **Cách làm:** cho mỗi tác giả 5, 10, 20, 35, 50 bài mẫu, lặp 20 lần.
- **Kết quả:** 20 bài thì power đạt 77%. Từ 20 lên 50 bài chỉ tăng thêm 7%.
- **Ý nghĩa:** nên có **ít nhất 20 bài mẫu** cho mỗi tác giả.

### TN4: Quét cả kho 500 bài thì sao?
- **Vấn đề:** kiểm tra 500 bài, mỗi bài có 5% khả năng báo nhầm, thì ước chừng có 25 bài bị nghi oan.
- **Cách làm:** so hai cách. Cách 1 là đánh dấu mọi bài có p ≤ 0,05. Cách 2 là dùng **Benjamini–Hochberg (BH)**, một thủ tục chỉnh ngưỡng khi kiểm định nhiều bài cùng lúc.
- **Kết quả:** khi kho có 5% bài sai, BH nâng **precision** (tỷ lệ cảnh báo đúng) từ **37% lên 64%**. Đổi lại, BH bỏ sót nhiều hơn.

### TN5: Bài chỉ bị viết thay **một phần**?
- **Cách làm:** thay một đoạn ngẫu nhiên trong bài của A bằng văn của B, với tỷ lệ 10%, 20%… đến 100%.
- **Kết quả:** chèn dưới 30% thì gần như không phát hiện được. Ở 70% mới phát hiện được 53%.
- **Bonus:** dùng "cửa sổ trượt" quét dọc bài để tìm **chỗ văn phong đổi**. Vị trí đoán được là từ thứ 390, vị trí thật là 400.

### TN6: Dùng máy sinh văn bản giả có lừa được không?
- **Cách làm:** dùng **chuỗi Markov** (mỗi từ tiếp theo chỉ phụ thuộc từ ngay trước) học từ bài của A, rồi sinh ra văn bản giả.
- **Kết quả:** **100%** văn bản giả qua mặt kiểm định, dù đọc thì vô nghĩa.
- **Ý nghĩa:** phương pháp chỉ nhìn tần suất từ, không kiểm tra câu có nghĩa hay không. Đây là **giới hạn** của phương pháp.

### TN7: Lặp bootstrap bao nhiêu lần là đủ?
- **Kết quả:** với 50 lần lặp, p-value dao động ±0,033, gần bằng chính ngưỡng 0,05, nên kết luận không ổn định. Với **1.000 lần** thì sai số còn ±0,007, đủ tin cậy. Kết quả thực nghiệm khớp với công thức lý thuyết √(p(1−p)/B).

### TN8: Dùng bao nhiêu từ đặc trưng?
- **Kết quả:** 300 từ cho AUC 0,947. Dùng nhiều từ hơn chỉ tăng chút ít và bắt đầu lẫn các từ chủ đề (tên công ty, ngành), vốn không phản ánh văn phong.

---

## 8. Kết luận cần nhớ

1. **Mô phỏng đúng điều kiện là quan trọng nhất.** Khớp độ dài giúp kéo báo động nhầm từ 48% xuống dưới 9%.
2. **Cần đủ dữ liệu:** bài từ 300 từ trở lên, tác giả có từ 20 bài mẫu trở lên.
3. **Quét cả kho thì dùng Benjamini–Hochberg.**
4. **Giới hạn:** văn phong thay đổi theo thời gian, bài chỉ bị viết thay một phần nhỏ, và văn bản giả bằng Markov.
5. Phương pháp phù hợp làm **công cụ sàng lọc** cho biên tập viên, không thay được con người.

---

## 9. Bảng thuật ngữ

| Thuật ngữ | Giải thích ngắn |
|---|---|
| NLP | Xử lý ngôn ngữ tự nhiên: cho máy tính làm việc với chữ |
| Stylometry | Phân tích văn phong bằng thống kê |
| Hư từ / MFW | Các từ phổ biến nhất (the, of, and…), ít phụ thuộc chủ đề |
| z-score | Chuẩn hoá để từ hiếm và từ phổ biến có trọng số ngang nhau |
| Cosine Delta | Khoảng cách văn phong, đo bằng góc giữa hai dãy số |
| Monte Carlo | Dùng lặp ngẫu nhiên rất nhiều lần để ước lượng một đại lượng |
| Bootstrap | Lấy mẫu lại có hoàn lại từ chính dữ liệu đang có |
| Phân phối null | Các giá trị thống kê sẽ có nếu H0 đúng |
| p-value | Xác suất gặp kết quả "lạ" như vậy nếu H0 đúng |
| Sai lầm loại I | Bác bỏ nhầm H0 (nghi oan bài đúng) |
| Power | Xác suất phát hiện được bài sai |
| AUC | Điểm phân biệt đúng/sai từ 0,5 (đoán mò) đến 1 (hoàn hảo) |
| Precision / Recall | Tỷ lệ cảnh báo đúng / tỷ lệ bài sai bắt được |
| FDR, Benjamini–Hochberg | Kiểm soát tỷ lệ cảnh báo nhầm khi kiểm định nhiều bài |
| Chuỗi Markov | Mô hình mà bước sau chỉ phụ thuộc bước ngay trước |
| Seed | "Hạt giống" ngẫu nhiên: cố định nó để chạy lại ra đúng kết quả |

---

## 10. Các file trong thư mục

| File / thư mục | Là gì |
|---|---|
| `README.md` | File bạn đang đọc |
| `DoAn_MoPhong.ipynb` | **Notebook chính**: chạy cả 9 thí nghiệm, vẽ hình, lưu kết quả (khoảng 2 phút) |
| `mophong.py` | Code lõi: đặc trưng văn phong, Cosine Delta, bootstrap, BH, chuỗi Markov |
| `thi_nghiem.py` | Nội dung giống notebook, dạng script |
| `app_demo.py` | App demo Streamlit |
| `chay_demo.bat` | **Bấm đúp để mở demo** |
| `data/` | Bộ dữ liệu Reuter_50_50 (50 nhà báo, 5.000 bài), dạng file zip, tự giải nén khi chạy. Nguồn: [UCI](https://archive.ics.uci.edu/dataset/217/reuter+50+50), giấy phép CC BY 4.0 |
| `figures/` | 10 hình kết quả, đã dùng trong báo cáo và slide |
| `results/ket_qua.json` | Toàn bộ con số kết quả |
| `HUONG_DAN.md` | Hướng dẫn chạy ngắn gọn |

### Tải code về máy
```
git clone <địa chỉ repo này>
```
hoặc bấm **Code → Download ZIP** trên GitHub rồi giải nén.

### Cách chạy notebook
1. Mở **Anaconda Navigator** → **Jupyter Notebook** (hoặc mở file bằng VS Code).
2. Mở `DoAn_MoPhong.ipynb`.
3. Chọn **Kernel → Restart & Run All**. Đợi khoảng 2 phút.

### Cách chạy demo
1. Bấm đúp `chay_demo.bat`.
2. Mở trình duyệt, vào **http://localhost:8501**. Lần đầu cần đợi khoảng 1,5 phút.
3. Chọn một bài test, chọn tên tác giả (thử cả đúng lẫn sai), rồi bấm **Kiểm định**.
4. Đóng cửa sổ đen để tắt app.

**Hai ví dụ chạy thử hay để demo:**
- Bài số 0, ghi tên **AaronPressman** (đúng tác giả): p ≈ 0,08 → ✅ văn phong phù hợp
- Bài số 0, ghi tên **DavidLawder** (sai tác giả): p ≈ 0,001 → ⚠️ nghi ngờ ghi sai, và bảng "5 tác giả gần nhất" chỉ ra đúng AaronPressman

---

## 11. Câu hỏi vấn đáp hay gặp và gợi ý trả lời

**"Mô phỏng nằm ở đâu trong đề tài này?"**
Ở hai tầng. Tầng 1: bootstrap tạo phân phối null để tính p-value. Tầng 2: các thí nghiệm Monte Carlo tạo tình huống biết trước đáp án để đánh giá phương pháp (mục 6).

**"Sao không dùng phân phối chuẩn cho nhanh?"**
Vì phân phối của khoảng cách bị lệch, không đối xứng và phụ thuộc độ dài bài. TN2 cho thấy dùng sai phân phối thì báo nhầm tới 48%.

**"Báo động nhầm 7,7% là sai rồi còn gì?"**
Em đã kiểm tra ở TN1b. Với bài cùng giai đoạn với bài mẫu, tỷ lệ chỉ 3,5%, đúng yêu cầu. Phần dư đến từ văn phong thay đổi theo thời gian, không phải lỗi thuật toán.

**"Sao chọn 1.000 lần lặp?"**
TN7: sai số p-value theo lý thuyết là √(p(1−p)/B). Với B = 1.000 thì sai số khoảng 0,007, đủ nhỏ so với ngưỡng 0,05.

**"Sao chọn 300 từ?"**
TN8: AUC tăng mạnh đến khoảng 300 rồi chững lại. Dùng nhiều từ hơn sẽ lẫn từ chủ đề vào đặc trưng.

**"Bài ghi sai lấy ở đâu ra?"**
Em tự tạo bằng mô phỏng: lấy bài của tác giả B, gán tên tác giả A (chọn ngẫu nhiên). Nhờ vậy biết trước đáp án để chấm điểm.

**"Biểu đồ null trong demo sao có nhiều đỉnh?"**
Bộ dữ liệu có bài gần như trùng nhau (bản cập nhật của cùng một tin). Khi bài trùng rơi vào hồ sơ, khoảng cách rất nhỏ, tạo ra một đỉnh ở bên trái. Kết luận của kiểm định không bị ảnh hưởng.

**"Áp dụng cho tiếng Việt được không?"**
Được, nhưng cần tách từ ghép (ví dụ bằng thư viện underthesea) và chọn lại danh sách hư từ tiếng Việt (thì, là, mà, của, đã, cũng…).

---

## 12. Nên đọc gì tiếp theo

1. **File này**: nắm ý tưởng.
2. **Slide**: mỗi slide có phần ghi chú lời nói, đọc như một kịch bản thuyết trình.
3. **Báo cáo, Chương 4 (Thiết kế mô phỏng) và Chương 5 (Kết quả)**: hai phần giảng viên hay hỏi nhất.
4. Mở notebook, chạy thử, và xem từng hình trong `figures/`.
"# mpnn" 
