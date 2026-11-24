# Hướng dẫn chạy đồ án

## Cấu trúc thư mục
- `data/` – đặt file `reuter+50+50.zip` (notebook tự giải nén)
- `mophong.py` – thư viện lõi: đặc trưng văn phong, Cosine Delta, kiểm định bootstrap, Benjamini–Hochberg, chuỗi Markov
- `DoAn_MoPhong.ipynb` – notebook chạy toàn bộ 9 thí nghiệm (TN1, TN1b, TN2–TN8)
- `thi_nghiem.py` – cùng nội dung notebook, dạng script
- `app_demo.py` – demo Streamlit
- `figures/` – các hình dùng trong báo cáo và slide
- `results/ket_qua.json` – toàn bộ số liệu kết quả; `results/nhat_ky_chay.txt` – nhật ký lần chạy

## Chạy notebook (Anaconda)
1. Mở Anaconda Prompt, `cd` tới thư mục này
2. `jupyter notebook DoAn_MoPhong.ipynb`
3. Kernel → Restart & Run All (khoảng 2–5 phút)

Cần: numpy, scipy, matplotlib (có sẵn trong Anaconda).

## Chạy demo
```
pip install streamlit      (nếu chưa có)
streamlit run app_demo.py
```
Lần đầu mở app sẽ mất ~30 giây để dựng hồ sơ 50 tác giả.
Lưu ý: chạy notebook ít nhất một lần trước để giải nén dữ liệu vào `data/`.
