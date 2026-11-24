"""
Demo: Kiểm tra bài báo có bị ghi sai tác giả không.
Chạy:  streamlit run app_demo.py
(cần thư mục data/C50train và data/C50test của bộ Reuter_50_50)
"""
import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from mophong import (load_c50, build_corpus, tokenize, rel_freq, zscore, dist,
                     AuthorProfile, mc_pvalue)

st.set_page_config(page_title="Phát hiện bài báo sai tác giả", page_icon="📰", layout="centered")
DATA = os.environ.get("C50_DIR", "data")
# Tự giải nén nếu trong data/ mới chỉ có file .zip
if not os.path.isdir(os.path.join(DATA, "C50train")):
    import glob, zipfile
    for z in glob.glob(os.path.join(DATA, "*.zip")):
        with zipfile.ZipFile(z) as f:
            f.extractall(DATA)


@st.cache_resource(show_spinner="Đang đọc kho bài báo và dựng hồ sơ văn phong…")
def setup(K=300):
    data = load_c50(DATA)
    C = build_corpus(data, K=K)
    rng = np.random.default_rng(2026)
    Z = np.vstack([zscore(C, rel_freq(d, K)) for d in C.train_ids])
    profs = []
    for a in range(len(C.authors)):
        idx = np.where(C.train_lab == a)[0]
        profs.append(AuthorProfile(C, [C.train_ids[i] for i in idx], rng, Zfull=Z[idx]))
    cen = np.vstack([p.centroid for p in profs])
    return data, C, profs, cen


data, C, profs, cen = setup()
w2i = {w: i for i, w in enumerate(C.vocab)}

st.title("📰 Bài báo này có đúng tác giả không?")
st.caption("Kiểm định bootstrap trên Cosine Delta – đồ án Mô phỏng ngẫu nhiên")

mode = st.radio("Nguồn văn bản", ["Lấy một bài test có sẵn", "Tự dán văn bản tiếng Anh"], horizontal=True)
if mode == "Lấy một bài test có sẵn":
    i = st.number_input("Chỉ số bài test (0–2499)", 0, len(C.test_ids) - 1, 0)
    true_author = C.authors[C.test_lab[i]]
    text_tokens = data["C50test"][i][1]
    st.text_area("Nội dung (300 từ đầu)", " ".join(text_tokens[:300]), height=150, disabled=True)
    st.info(f"Tác giả thật của bài này: **{true_author}**")
else:
    txt = st.text_area("Dán bài báo (tiếng Anh, nên ≥ 300 từ)", height=200)
    text_tokens = tokenize(txt)

claimed = st.selectbox("Tác giả được ghi trên bài", C.authors)
B = st.select_slider("Số lần lặp bootstrap B", [200, 500, 1000, 2000], value=1000)
alpha = st.select_slider("Mức ý nghĩa α", [0.01, 0.05, 0.10], value=0.05)

if st.button("Kiểm định", type="primary") and len(text_tokens) > 0:
    ids = np.array([w2i.get(t, -1) for t in text_tokens])
    L = len(ids)
    a = C.authors.index(claimed)
    z = zscore(C, rel_freq(ids, C.K))
    T = float(dist(z, cen[a]))
    null = profs[a].null(L if L < 450 else None, B)
    p = float(mc_pvalue(np.array([T]), null)[0])
    se = np.sqrt(p * (1 - p) / B)

    c1, c2, c3 = st.columns(3)
    c1.metric("Số từ", L)
    c2.metric("Cosine Delta quan sát", f"{T:.3f}")
    c3.metric("p-value", f"{p:.4f}", help=f"Sai số Monte Carlo ≈ {se:.4f}")
    if p <= alpha:
        st.error(f"⚠️ Bác bỏ H0 ở mức α = {alpha}: văn phong **không khớp** với {claimed} – nghi ngờ ghi sai tác giả.")
    else:
        st.success(f"✅ Chưa đủ bằng chứng bác bỏ: văn phong phù hợp với {claimed}.")
    if L < 200:
        st.warning("Văn bản ngắn hơn 200 từ – kết luận kém tin cậy (xem TN2 trong báo cáo).")

    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.hist(null, bins=40, color="#2a78d6", edgecolor="white")
    ax.axvline(T, color="#eb6834", lw=2.5)
    ax.text(T, ax.get_ylim()[1] * 0.9, "  bài đang xét", color="#eb6834")
    ax.set_xlabel("Khoảng cách dưới H0 (mô phỏng bootstrap)")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)

    dall = dist(z[None], cen)
    top = np.argsort(dall)[:5]
    st.subheader("5 tác giả có văn phong gần nhất")
    st.table({"Tác giả": [C.authors[t] for t in top], "Khoảng cách": [f"{dall[t]:.3f}" for t in top]})
