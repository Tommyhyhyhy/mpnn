# %% [markdown]
# # Đồ án Mô phỏng ngẫu nhiên
# ## Phát hiện bài báo bị ghi sai tác giả bằng NLP
#
# Notebook này chạy toàn bộ thí nghiệm của đồ án. Chạy lần lượt từ trên xuống
# (Kernel → Restart & Run All). Thời gian chạy khoảng 5–15 phút tuỳ máy.
#
# **Dữ liệu:** bộ Reuter_50_50 (UCI) – 50 nhà báo Reuters, mỗi người 50 bài
# train + 50 bài test. Giải nén vào thư mục `data/` sao cho có
# `data/C50train/` và `data/C50test/`.
#
# **Các thí nghiệm mô phỏng**
# | Mã | Nội dung |
# |---|---|
# | TN1 | Hiệu chuẩn kiểm định: sai lầm loại I, phân phối p-value, power, ROC |
# | TN1b | Trôi văn phong theo thời gian: vì sao sai lầm loại I > 5% |
# | TN2 | Ảnh hưởng của độ dài bài báo L |
# | TN3 | Ảnh hưởng của số bài mẫu n của tác giả (Monte Carlo lặp R lần) |
# | TN4 | Mô phỏng kho bài bị gán nhãn sai p% + hiệu chỉnh Benjamini–Hochberg |
# | TN5 | Bài viết pha trộn (ghostwriting một phần) + phát hiện điểm chuyển |
# | TN6 | Văn bản giả sinh bằng chuỗi Markov – tấn công giả mạo văn phong |
# | TN7 | Sai số Monte Carlo: chọn số lần lặp bootstrap B |
# | TN8 | Độ nhạy theo số từ đặc trưng K |

# %%
import os, json, time, zipfile, glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

from mophong import (load_c50, build_corpus, rel_freq, zscore, delta, dist,
                     random_chunk, AuthorProfile, mc_pvalue,
                     benjamini_hochberg, BigramMarkov)

SEED = 2026
rng = np.random.default_rng(SEED)
DATA = os.environ.get("C50_DIR", "data")
FIG, RES = "figures", "results"
os.makedirs(FIG, exist_ok=True); os.makedirs(RES, exist_ok=True)
FAST = os.environ.get("FAST", "0") == "1"   # chế độ chạy nhanh để thử code

# Tự giải nén nếu chỉ có file .zip trong data/
if not os.path.isdir(os.path.join(DATA, "C50train")):
    for z in glob.glob(os.path.join(DATA, "*.zip")):
        with zipfile.ZipFile(z) as f:
            f.extractall(DATA)
    # một số bản zip lồng thêm C50.zip bên trong
    for z in glob.glob(os.path.join(DATA, "**", "*.zip"), recursive=True):
        if not os.path.isdir(os.path.join(DATA, "C50train")):
            with zipfile.ZipFile(z) as f:
                f.extractall(DATA)
    found = glob.glob(os.path.join(DATA, "**", "C50train"), recursive=True)
    if found:
        DATA = os.path.dirname(found[0])

# Màu (bảng màu đã kiểm tra mù màu): xanh dương, cam, xanh ngọc
C1, C2, C3, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "axes.edgecolor": "#8a8984", "axes.titleweight": "bold",
    "lines.linewidth": 2, "legend.frameon": False,
})
RESULTS = {}

def wilson(k, n, z=1.96):
    """Khoảng tin cậy Wilson 95% cho tỷ lệ k/n."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)

def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.show() if matplotlib.get_backend().lower() not in ("agg",) else None
    plt.close()

# %% [markdown]
# ## 1. Đọc dữ liệu và xây dựng đặc trưng văn phong

# %%
t0 = time.time()
data = load_c50(DATA)
K = 300
C = build_corpus(data, K=K)
A = len(C.authors)
raw_train = [t for _, t in data["C50train"]]
raw_test = [t for _, t in data["C50test"]]
lens = np.array([len(t) for t in raw_train + raw_test])
print(f"{A} tác giả | {len(C.train_ids)} bài train | {len(C.test_ids)} bài test")
print(f"Độ dài bài: trung vị {np.median(lens):.0f} từ, min {lens.min()}, max {lens.max()}")
print("20 từ đặc trưng đầu tiên:", C.vocab[:20])
RESULTS["data"] = dict(n_authors=A, n_train=len(C.train_ids), n_test=len(C.test_ids),
                       len_median=float(np.median(lens)), len_min=int(lens.min()),
                       len_max=int(lens.max()), len_mean=float(lens.mean()),
                       K=K, top_words=C.vocab[:30])

fig, ax = plt.subplots(figsize=(6, 3))
ax.hist(lens, bins=40, color=C1, edgecolor="white", linewidth=0.8)
ax.set_xlabel("Số từ trong bài"); ax.set_ylabel("Số bài")
ax.set_title("Phân bố độ dài bài báo (Reuter_50_50)")
savefig("00_do_dai_bai.png")

# %% [markdown]
# ## 2. Hồ sơ tác giả và phân phối null mô phỏng
# Với mỗi tác giả ta dựng centroid từ 50 bài train và mô phỏng B giá trị
# Cosine Delta dưới H0 bằng bootstrap (xem `mophong.AuthorProfile.null`).

# %%
B = 300 if FAST else 1000
by_author = [[i for i in range(len(C.train_ids)) if C.train_lab[i] == a] for a in range(A)]
Ztrain = np.vstack([zscore(C, rel_freq(d, K)) for d in C.train_ids])
profiles = [AuthorProfile(C, [C.train_ids[i] for i in idx], rng, Zfull=Ztrain[idx])
            for idx in by_author]
centroids = np.vstack([p.centroid for p in profiles])          # (A, K)
nulls_full = [p.null(None, B) for p in profiles]
Ztest = np.vstack([zscore(C, rel_freq(d, K)) for d in C.test_ids])
D_test = dist(Ztest[:, None, :], centroids[None, :, :])       # (n_test, A)
P_test = np.column_stack([mc_pvalue(D_test[:, a], nulls_full[a]) for a in range(A)])
print(f"Đã mô phỏng null cho {A} tác giả, B = {B}  ({time.time()-t0:.0f}s)")

a0 = 0
fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.hist(nulls_full[a0], bins=40, density=True, color=C1, alpha=0.85,
        edgecolor="white", linewidth=0.6, label="Null mô phỏng (H0 đúng)")
own = D_test[C.test_lab == a0, a0]
oth = D_test[C.test_lab != a0, a0]
ax.hist(own, bins=25, density=True, histtype="step", color=C3, linewidth=2,
        label="Bài test thật của tác giả")
ax.hist(oth, bins=40, density=True, histtype="step", color=C2, linewidth=2,
        label="Bài của tác giả khác")
ax.axvline(np.quantile(nulls_full[a0], 0.95), color=GRAY, ls="--", lw=1.2)
ax.text(np.quantile(nulls_full[a0], 0.95), ax.get_ylim()[1] * 0.92, " ngưỡng 5%",
        color="#52514e", fontsize=9)
ax.set_xlabel("Cosine Delta tới hồ sơ tác giả")
ax.set_ylabel("Mật độ")
ax.set_title(f"Phân phối khoảng cách – tác giả {C.authors[a0]}")
ax.legend(fontsize=8)
savefig("01_null_mot_tac_gia.png")

# %% [markdown]
# ## TN1. Hiệu chuẩn kiểm định và năng lực phát hiện (bài đầy đủ)
# * Bài thật được ghi đúng tác giả → p-value nên ~ Uniform(0,1), tỷ lệ bác bỏ ≈ α.
# * Bài bị ghi sai sang tác giả khác (chọn ngẫu nhiên) → p-value nên nhỏ.

# %%
alpha = 0.05
n_te = len(C.test_ids)
p_true = P_test[np.arange(n_te), C.test_lab]
fake_claim = np.array([rng.choice([a for a in range(A) if a != t]) for t in C.test_lab])
p_false = P_test[np.arange(n_te), fake_claim]
t1 = float(np.mean(p_true <= alpha)); pw = float(np.mean(p_false <= alpha))
ks = stats.kstest(p_true, "uniform")

# ROC: dùng 1 - p làm điểm số nghi ngờ
scores = np.r_[1 - p_true, 1 - p_false]; y = np.r_[np.zeros(n_te), np.ones(n_te)]
order = np.argsort(-scores, kind="mergesort")
tpr = np.r_[0, np.cumsum(y[order]) / y.sum()]
fpr = np.r_[0, np.cumsum(1 - y[order]) / (1 - y).sum()]
_trap = getattr(np, "trapezoid", None) or np.trapz   # numpy 1.x và 2.x
auc = float(_trap(tpr, fpr))
# Nhận diện tác giả thật bằng Delta nhỏ nhất (bài toán phụ)
acc_attr = float(np.mean(D_test.argmin(1) == C.test_lab))
print(f"Sai lầm loại I (α=5%): {t1:.3f}  CI95 {wilson(t1*n_te, n_te)}")
print(f"Power phát hiện bài sai tác giả: {pw:.3f}  CI95 {wilson(pw*n_te, n_te)}")
print(f"KS test p_true ~ U(0,1): D={ks.statistic:.3f}, p={ks.pvalue:.2e}")
print(f"AUC = {auc:.3f} | Độ chính xác đoán đúng tác giả thật (argmin Delta): {acc_attr:.3f}")
# Mốc so sánh: cùng quy trình nhưng dùng Delta cổ điển của Burrows
prof_cl = [AuthorProfile(C, p.docs, rng, Zfull=p.Z, metric=delta) for p in profiles]
null_cl = [p.null(None, B) for p in prof_cl]
D_cl = delta(Ztest[:, None, :], centroids[None, :, :])
pt_cl = np.array([mc_pvalue(D_cl[i:i+1, C.test_lab[i]], null_cl[C.test_lab[i]])[0] for i in range(n_te)])
pf_cl = np.array([mc_pvalue(D_cl[i:i+1, fake_claim[i]], null_cl[fake_claim[i]])[0] for i in range(n_te)])
auc_cl = float(stats.mannwhitneyu(1 - pf_cl, 1 - pt_cl).statistic / (n_te * n_te))
baseline = dict(type1=float(np.mean(pt_cl <= alpha)), power=float(np.mean(pf_cl <= alpha)), auc=auc_cl,
                attribution_acc=float(np.mean(D_cl.argmin(1) == C.test_lab)))
print("Mốc so sánh Delta cổ điển:", baseline)
RESULTS["tn1_baseline_classic_delta"] = baseline
RESULTS["tn1"] = dict(alpha=alpha, B=B, type1=t1, type1_ci=wilson(t1*n_te, n_te),
                      power=pw, power_ci=wilson(pw*n_te, n_te), ks_D=float(ks.statistic),
                      ks_p=float(ks.pvalue), auc=auc, attribution_acc=acc_attr,
                      p_true_median=float(np.median(p_true)))

fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
bins = np.linspace(0, 1, 21)
axs[0].hist(p_true, bins=bins, color=C3, edgecolor="white", linewidth=0.8, label="Ghi đúng tác giả")
axs[0].hist(p_false, bins=bins, color=C2, alpha=0.75, edgecolor="white", linewidth=0.8, label="Ghi sai tác giả")
axs[0].axhline(n_te / 20, color=GRAY, ls="--", lw=1.2)
axs[0].text(0.5, n_te / 20, "mức kỳ vọng nếu Uniform", color="#52514e", fontsize=8, va="bottom", ha="center")
axs[0].set_xlabel("p-value Monte Carlo"); axs[0].set_ylabel("Số bài"); axs[0].legend(fontsize=8)
axs[0].set_title("Phân phối p-value")
sc2 = np.r_[1 - pt_cl, 1 - pf_cl]; o2 = np.argsort(-sc2, kind="mergesort")
tpr2 = np.r_[0, np.cumsum(y[o2]) / y.sum()]; fpr2 = np.r_[0, np.cumsum(1 - y[o2]) / (1 - y).sum()]
axs[1].plot(fpr, tpr, color=C1, label=f"Cosine Delta (AUC {auc:.3f})")
axs[1].plot(fpr2, tpr2, color=GRAY, lw=1.5, label=f"Delta cổ điển (AUC {auc_cl:.3f})")
axs[1].plot([0, 1], [0, 1], color=GRAY, ls=":", lw=1)
axs[1].legend(fontsize=8, loc="lower right")
axs[1].set_xlabel("Tỷ lệ báo động nhầm (FPR)"); axs[1].set_ylabel("Tỷ lệ phát hiện (TPR)")
axs[1].set_title("Đường ROC")
savefig("02_tn1_pvalue_roc.png")

# %% [markdown]
# ## TN1b. Vì sao sai lầm loại I cao hơn 5%? Kiểm tra "trôi văn phong" theo thời gian
# Chia 50 bài train của mỗi tác giả thành 25 bài dựng hồ sơ + 25 bài giữ lại
# (cùng giai đoạn). So sánh tỷ lệ báo động nhầm trên 25 bài giữ lại với
# 25 bài test (giai đoạn khác). Lặp 3 lần với cách chia ngẫu nhiên.

# %%
r_in, r_out = [], []
for rep in range(1 if FAST else 3):
    for a in range(A):
        idx = rng.permutation(by_author[a]); prof_i, hold = idx[:25], idx[25:]
        P = AuthorProfile(C, [C.train_ids[i] for i in prof_i], rng, Zfull=Ztrain[prof_i])
        nl = P.null(None, B)
        r_in += list(mc_pvalue(dist(Ztrain[hold], P.centroid), nl) <= alpha)
        te = rng.choice(np.where(C.test_lab == a)[0], 25, replace=False)
        r_out += list(mc_pvalue(dist(Ztest[te], P.centroid), nl) <= alpha)
drift = dict(same_period=float(np.mean(r_in)), same_period_ci=wilson(np.sum(r_in), len(r_in)),
             other_period=float(np.mean(r_out)), other_period_ci=wilson(np.sum(r_out), len(r_out)), n=len(r_in))
print(drift); RESULTS["tn1b_drift"] = drift
fig, ax = plt.subplots(figsize=(5.5, 3))
vals = [drift["same_period"], drift["other_period"]]
cis = [drift["same_period_ci"], drift["other_period_ci"]]
ax.bar([0, 1], vals, 0.5, color=[C3, C2],
       yerr=[[v - c[0] for v, c in zip(vals, cis)], [c[1] - v for v, c in zip(vals, cis)]],
       capsize=4, error_kw=dict(lw=1, ecolor="#52514e"))
for i, (v, c) in enumerate(zip(vals, cis)):
    ax.text(i, c[1] + 0.004, f"{v*100:.1f}%", ha="center", color="#0b0b0b")
ax.grid(axis="x", visible=False)
ax.axhline(alpha, color="#52514e", ls="--", lw=1); ax.text(0.5, alpha + 0.002, "α = 5%", ha="center", va="bottom", color="#52514e", fontsize=9)
ax.set_xticks([0, 1], ["Bài cùng giai đoạn\nvới bài mẫu", "Bài ở giai đoạn khác\n(tập test)"])
ax.set_ylabel("Tỷ lệ báo động nhầm"); ax.set_ylim(0, max(c[1] for c in cis) * 1.25)
ax.set_title("TN1b – Trôi văn phong làm tăng báo động nhầm")
savefig("02b_tn1b_troi_van_phong.png")

# %% [markdown]
# ## TN2. Ảnh hưởng của độ dài bài báo L
# Cắt ngẫu nhiên một đoạn L từ trong mỗi bài test; null cũng được mô phỏng với
# đúng độ dài L ("null khớp độ dài"). So sánh thêm với cách làm SAI là dùng
# null của bài đầy đủ cho bài ngắn.

# %%
Ls = [50, 100, 200, 300, 400] if not FAST else [100, 300]
B2 = 300 if FAST else 600
tn2 = []
for L in Ls:
    nullL = [p.null(L, B2) for p in profiles]
    Zc = np.vstack([zscore(C, rel_freq(random_chunk(d, L, rng), K)) for d in C.test_ids])
    DL = dist(Zc[:, None, :], centroids[None, :, :])
    pt = np.array([mc_pvalue(DL[i:i+1, C.test_lab[i]], nullL[C.test_lab[i]])[0] for i in range(n_te)])
    pf = np.array([mc_pvalue(DL[i:i+1, fake_claim[i]], nullL[fake_claim[i]])[0] for i in range(n_te)])
    pt_wrong = np.array([mc_pvalue(DL[i:i+1, C.test_lab[i]], nulls_full[C.test_lab[i]])[0] for i in range(n_te)])
    row = dict(L=L, type1=float(np.mean(pt <= alpha)), power=float(np.mean(pf <= alpha)),
               type1_wrongnull=float(np.mean(pt_wrong <= alpha)),
               acc_attr=float(np.mean(DL.argmin(1) == C.test_lab)))
    row["power_ci"] = wilson(row["power"] * n_te, n_te)
    row["type1_ci"] = wilson(row["type1"] * n_te, n_te)
    tn2.append(row); print(row)
full = dict(L="đầy đủ", type1=t1, power=pw, acc_attr=acc_attr)
RESULTS["tn2"] = tn2

fig, ax = plt.subplots(figsize=(6.5, 3.5))
x = [r["L"] for r in tn2]
ax.plot(x, [r["power"] for r in tn2], "o-", color=C2, ms=6, label="Power (phát hiện bài sai)")
ax.fill_between(x, [r["power_ci"][0] for r in tn2], [r["power_ci"][1] for r in tn2], color=C2, alpha=0.15, lw=0)
ax.plot(x, [r["type1"] for r in tn2], "o-", color=C3, ms=6, label="Sai lầm loại I – null khớp độ dài")
ax.plot(x, [r["type1_wrongnull"] for r in tn2], "s--", color=GRAY, ms=5, lw=1.5, label="Sai lầm loại I – null bài đầy đủ (sai)")
ax.axhline(alpha, color="#52514e", lw=0.8, ls=":")
ax.set_xlabel("Độ dài đoạn văn L (từ)"); ax.set_ylabel("Tỷ lệ")
ax.set_ylim(0, 1.02); ax.set_title("TN2 – Độ dài bài ảnh hưởng tới kiểm định")
ax.legend(fontsize=8, loc="center right")
savefig("03_tn2_do_dai.png")

# %% [markdown]
# ## TN3. Ảnh hưởng của số bài mẫu n (Monte Carlo lặp R lần)
# Mỗi lần lặp: rút ngẫu nhiên n bài train của mỗi tác giả để dựng hồ sơ + null,
# rồi kiểm định trên 10 bài test ngẫu nhiên mỗi tác giả.

# %%
ns = [5, 10, 20, 35, 50] if not FAST else [5, 20]
R3 = 4 if FAST else 20
B3 = 200 if FAST else 300
tn3 = []
for n in ns:
    t1s, pws = [], []
    for r in range(R3):
        sub = [rng.choice(idx, size=n, replace=False) for idx in by_author]
        prof = [AuthorProfile(C, [C.train_ids[i] for i in s], rng, Zfull=Ztrain[s]) for s in sub]
        cen = np.vstack([p.centroid for p in prof])
        nl = [p.null(None, B3) for p in prof]
        pick = np.concatenate([rng.choice(np.where(C.test_lab == a)[0], 10, replace=False) for a in range(A)])
        D = dist(Ztest[pick][:, None, :], cen[None])
        lab = C.test_lab[pick]
        fk = np.array([rng.choice([a for a in range(A) if a != t]) for t in lab])
        pt = np.array([mc_pvalue(D[i:i+1, lab[i]], nl[lab[i]])[0] for i in range(len(pick))])
        pf = np.array([mc_pvalue(D[i:i+1, fk[i]], nl[fk[i]])[0] for i in range(len(pick))])
        t1s.append(np.mean(pt <= alpha)); pws.append(np.mean(pf <= alpha))
    t1s, pws = np.array(t1s), np.array(pws)
    row = dict(n=n, R=R3, type1=float(t1s.mean()), type1_sd=float(t1s.std(ddof=1)),
               power=float(pws.mean()), power_sd=float(pws.std(ddof=1)))
    tn3.append(row); print(row)
RESULTS["tn3"] = tn3

fig, ax = plt.subplots(figsize=(6.5, 3.5))
x = [r["n"] for r in tn3]
for key, col, lab in [("power", C2, "Power"), ("type1", C3, "Sai lầm loại I")]:
    m = np.array([r[key] for r in tn3]); s = np.array([r[key + "_sd"] for r in tn3])
    ax.errorbar(x, m, yerr=1.96 * s / np.sqrt(R3), fmt="o-", color=col, ms=6, capsize=3, label=lab)
ax.axhline(alpha, color="#52514e", lw=0.8, ls=":")
ax.set_xlabel("Số bài mẫu của mỗi tác giả (n)"); ax.set_ylabel("Tỷ lệ (trung bình R lần lặp)")
ax.set_ylim(0, 1.02); ax.set_title(f"TN3 – Cần bao nhiêu bài mẫu? (R = {R3})")
ax.legend(fontsize=8)
savefig("04_tn3_so_bai_mau.png")

# %% [markdown]
# ## TN4. Mô phỏng kho bài báo bị gán nhãn sai p% + Benjamini–Hochberg
# Mỗi lần lặp: rút N = 500 bài test, chọn ngẫu nhiên p% bài và đổi nhãn sang
# tác giả khác. Kiểm định tất cả các bài, so sánh hai quy tắc:
# (a) bác bỏ khi p ≤ 0.05 cho từng bài; (b) Benjamini–Hochberg kiểm soát FDR 5%.

# %%
N4, R4 = 500, (30 if FAST else 300)
props = [0.02, 0.05, 0.10, 0.20]
tn4 = []
for prop in props:
    rec = {"naive": [], "bh": []}
    rec_attr = []
    for r in range(R4):
        idx = rng.choice(n_te, N4, replace=False)
        lab = C.test_lab[idx].copy()
        m = int(round(prop * N4))
        bad = rng.choice(N4, m, replace=False)
        claim = lab.copy()
        for j in bad:
            claim[j] = rng.choice([a for a in range(A) if a != lab[j]])
        isbad = np.zeros(N4, bool); isbad[bad] = True
        p = P_test[idx, claim]
        for name, flag in [("naive", p <= alpha), ("bh", benjamini_hochberg(p, 0.05))]:
            tp = np.sum(flag & isbad); fp = np.sum(flag & ~isbad); fn = np.sum(~flag & isbad)
            prec = tp / max(tp + fp, 1); recl = tp / max(tp + fn, 1)
            f1 = 2 * prec * recl / max(prec + recl, 1e-12)
            rec[name].append((prec, recl, f1, fp / max(tp + fp, 1), flag.sum()))
        flagged = np.where(benjamini_hochberg(p, 0.05) & isbad)[0]
        if len(flagged):
            rec_attr.append(np.mean(D_test[idx[flagged]].argmin(1) == lab[flagged]))
    row = dict(prop=prop, N=N4, R=R4)
    for name in rec:
        arr = np.array(rec[name])
        for k, nm in enumerate(["precision", "recall", "f1", "fdr", "n_flag"]):
            row[f"{name}_{nm}"] = float(arr[:, k].mean())
            row[f"{name}_{nm}_sd"] = float(arr[:, k].std(ddof=1))
    row["recover_true_author"] = float(np.mean(rec_attr)) if rec_attr else None
    tn4.append(row)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if not k.endswith("_sd")})
RESULTS["tn4"] = tn4

fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
x = np.arange(len(props)); w = 0.36
for ax, met, ttl in [(axs[0], "precision", "Precision (độ chính xác khi báo động)"),
                     (axs[1], "recall", "Recall (tỷ lệ bắt được bài sai)")]:
    for k, (name, col, lab) in enumerate([("naive", C1, "Ngưỡng p ≤ 0.05"), ("bh", C2, "Benjamini–Hochberg")]):
        m = [r[f"{name}_{met}"] for r in tn4]; s = [r[f"{name}_{met}_sd"] for r in tn4]
        ax.bar(x + (k - 0.5) * w, m, w * 0.92, color=col, label=lab, yerr=s, capsize=2,
               error_kw=dict(lw=0.8, ecolor="#52514e"))
    ax.set_xticks(x, [f"{int(p*100)}%" for p in props]); ax.set_ylim(0, 1.05)
    ax.set_xlabel("Tỷ lệ bài bị ghi sai trong kho"); ax.set_title(ttl, fontsize=10)
axs[0].legend(fontsize=8, loc="upper left")
savefig("05_tn4_kho_bai_BH.png")

# %% [markdown]
# ## TN5. Bài viết pha trộn: một phần bài được viết bởi người khác
# Lấy đoạn dài 400 từ của tác giả A, thay một đoạn liên tục chiếm tỷ lệ f
# (vị trí ngẫu nhiên) bằng văn bản của tác giả B. Kiểm định với tác giả A.

# %%
L5 = 400
fs = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
null400 = [p.null(L5, B2) for p in profiles]
long_te = np.array([i for i in range(n_te) if len(C.test_ids[i]) >= L5])
R5 = 300 if FAST else 2000
tn5 = []
def spliced(i, f, rng):
    a = C.test_lab[i]
    base = random_chunk(C.test_ids[i], L5, rng).copy()
    k = int(round(f * L5))
    if k == 0:
        return base, a
    b = rng.choice([x for x in range(A) if x != a])
    donor_i = rng.choice(np.where((C.test_lab == b))[0])
    donor = C.test_ids[donor_i]
    seg = random_chunk(donor, k, rng)
    k = len(seg)
    s = rng.integers(0, L5 - k + 1)
    base[s:s + k] = seg
    return base, a
for f in fs:
    ps = []
    for r in range(R5):
        i = rng.choice(long_te)
        ids, a = spliced(i, f, rng)
        T = dist(zscore(C, rel_freq(ids, K)), centroids[a])
        ps.append(mc_pvalue(np.array([T]), null400[a])[0])
    ps = np.array(ps)
    row = dict(f=f, reject=float(np.mean(ps <= alpha)), ci=wilson(np.sum(ps <= alpha), R5), R=R5)
    tn5.append(row); print(row)
RESULTS["tn5"] = tn5

# Minh hoạ phát hiện điểm chuyển văn phong bằng cửa sổ trượt
# Chọn bài minh hoạ "điển hình": bài của A được nhận đúng là của A (p lớn),
# bài của B cũng được nhận đúng là của B
good = [i for i in long_te if D_test[i].argmin() == C.test_lab[i] and p_true[i] > 0.3]
i_demo = rng.choice(good)
a_demo = C.test_lab[i_demo]
j_demo = rng.choice([j for j in good if C.test_lab[j] != a_demo])
b_demo = C.test_lab[j_demo]
donor_demo = C.test_ids[j_demo]
part1 = C.test_ids[i_demo][:400]; part2 = donor_demo[:400]
mix = np.r_[part1, part2]
W, S = 200, 10
pos, dA, dB = [], [], []
for s in range(0, len(mix) - W + 1, S):
    z = zscore(C, rel_freq(mix[s:s + W], K))
    pos.append(s + W / 2); dA.append(dist(z, centroids[a_demo])); dB.append(dist(z, centroids[b_demo]))
pos, dA, dB = map(np.array, (pos, dA, dB))
# change-point: tách đôi chuỗi chênh lệch (dA - dB) sao cho tổng bình phương sai số nhỏ nhất
g = dA - dB
cost = [np.var(g[:k]) * k + np.var(g[k:]) * (len(g) - k) for k in range(3, len(g) - 3)]
kstar = int(np.argmin(cost)) + 3
cp_est = float(pos[kstar])
RESULTS["tn5_changepoint"] = dict(true=400, est=cp_est, window=W, author_A=C.authors[a_demo],
                                  author_B=C.authors[b_demo])
print("Điểm chuyển thật = 400, ước lượng =", cp_est)

fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.4))
axs[0].plot([r["f"] * 100 for r in tn5], [r["reject"] for r in tn5], "o-", color=C2, ms=6)
axs[0].fill_between([r["f"] * 100 for r in tn5], [r["ci"][0] for r in tn5], [r["ci"][1] for r in tn5], color=C2, alpha=0.15, lw=0)
axs[0].axhline(alpha, color="#52514e", lw=0.8, ls=":")
axs[0].set_xlabel("Tỷ lệ văn bản do người khác viết f (%)"); axs[0].set_ylabel("Tỷ lệ bị phát hiện")
axs[0].set_ylim(0, 1.02); axs[0].set_title("TN5 – Phát hiện bài pha trộn (L = 400)")
axs[1].plot(pos, dA, color=C1, label=f"Khoảng cách tới A ({C.authors[a_demo]})")
axs[1].plot(pos, dB, color=C2, label=f"Khoảng cách tới B ({C.authors[b_demo]})")
axs[1].axvline(400, color="#52514e", lw=1, ls=":")
axs[1].axvline(cp_est, color=C3, lw=1.5, ls="--")
axs[1].text(400, axs[1].get_ylim()[1], " điểm chuyển thật", fontsize=8, va="top", color="#52514e")
axs[1].text(cp_est, axs[1].get_ylim()[0], f" ước lượng = {cp_est:.0f}", fontsize=8, va="bottom", color=C3)
axs[1].set_xlabel("Vị trí trong văn bản (từ)"); axs[1].set_ylabel("Cosine Delta (cửa sổ 200 từ)")
axs[1].set_title("Phát hiện điểm chuyển văn phong"); axs[1].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
savefig("06_tn5_pha_tron.png")

# %% [markdown]
# ## TN6. Văn bản giả mạo sinh bằng chuỗi Markov
# Học chuỗi Markov bigram từ các bài train của tác giả A, sinh văn bản 500 từ
# rồi kiểm định "văn bản này có phải của A?". Nếu kiểm định chấp nhận → kẻ giả
# mạo đã qua mặt được phương pháp dựa trên tần suất hư từ.

# %%
L6 = 500
n_gen = 5 if FAST else 20
w2i = {w: i for i, w in enumerate(C.vocab)}
null500 = [p.null(L6, B2) for p in profiles]
acc_self, acc_other, sample_text = [], [], None
authors_train_tok = [[raw_train[i] for i in idx] for idx in by_author]
for a in range(A if not FAST else 10):
    mk = BigramMarkov(authors_train_tok[a])
    for g_ in range(n_gen):
        toks = mk.sample(L6, rng)
        if sample_text is None:
            sample_text = " ".join(toks[:60])
        ids = np.array([w2i.get(t, -1) for t in toks])
        z = zscore(C, rel_freq(ids, K))
        ps = mc_pvalue(np.array([dist(z, centroids[a])]), null500[a])[0]
        b = rng.choice([x for x in range(A) if x != a])
        po = mc_pvalue(np.array([dist(z, centroids[b])]), null500[b])[0]
        acc_self.append(ps > alpha); acc_other.append(po > alpha)
RESULTS["tn6"] = dict(L=L6, n=len(acc_self), pass_as_author=float(np.mean(acc_self)),
                      pass_as_other=float(np.mean(acc_other)), sample=sample_text)
print(RESULTS["tn6"])

# %% [markdown]
# ## TN7. Sai số Monte Carlo: chọn số lần lặp bootstrap B
# p̂ là một ước lượng Monte Carlo nên có sai số chuẩn ≈ sqrt(p(1−p)/B).
# Lặp 40 lần với mỗi B để đo độ dao động thực nghiệm.

# %%
Bs = [50, 100, 200, 500, 1000, 2000] if not FAST else [50, 200]
# chọn một bài có p-value "giáp ranh" (gần 0.05) để thấy rõ ảnh hưởng của B
i7 = int(np.argmin(np.abs(p_true - 0.05)))
a7 = C.test_lab[i7]
T7 = D_test[i7, a7]
tn7 = []
for Bb in Bs:
    est = []
    for r in range(40 if not FAST else 5):
        pr = AuthorProfile(C, profiles[a7].docs, np.random.default_rng(10_000 + r * 7 + Bb), Zfull=profiles[a7].Z)
        est.append(mc_pvalue(np.array([T7]), pr.null(None, Bb))[0])
    est = np.array(est)
    tn7.append(dict(B=Bb, mean=float(est.mean()), sd=float(est.std(ddof=1)),
                    sd_theory=float(np.sqrt(est.mean() * (1 - est.mean()) / Bb))))
    print(tn7[-1])
RESULTS["tn7"] = tn7

fig, ax = plt.subplots(figsize=(6.5, 3.4))
xb = [r["B"] for r in tn7]
ax.plot(xb, [r["sd"] for r in tn7], "o-", color=C1, ms=6, label="Độ lệch chuẩn thực nghiệm của p̂")
ax.plot(xb, [r["sd_theory"] for r in tn7], "--", color=C2, label="Lý thuyết √(p(1−p)/B)")
ax.set_xscale("log"); ax.set_xticks(xb, [str(b) for b in xb]); ax.minorticks_off(); ax.set_xlabel("Số lần lặp bootstrap B (thang log)")
ax.set_ylabel("Sai số chuẩn của p̂"); ax.set_title("TN7 – Sai số Monte Carlo giảm theo B")
ax.legend(fontsize=8)
savefig("07_tn7_sai_so_MC.png")

# %% [markdown]
# ## TN8. Độ nhạy theo số từ đặc trưng K
# Tính AUC phân biệt "ghi đúng" và "ghi sai" trực tiếp từ khoảng cách (không cần null),
# so sánh Cosine Delta với Delta cổ điển.

# %%
Ks = [25, 50, 100, 200, 300, 500, 1000] if not FAST else [50, 300]
tn8 = []
for Kk in Ks:
    Ck = build_corpus(data, K=Kk)
    Ztr = np.vstack([zscore(Ck, rel_freq(d, Kk)) for d in Ck.train_ids])
    cen = np.vstack([Ztr[Ck.train_lab == a].mean(0) for a in range(A)])
    Zte = np.vstack([zscore(Ck, rel_freq(d, Kk)) for d in Ck.test_ids])
    row = dict(K=Kk)
    for nm, fn in [("cos", dist), ("classic", delta)]:
        D = fn(Zte[:, None, :], cen[None])
        s_true = D[np.arange(n_te), Ck.test_lab]; s_false = D[np.arange(n_te), fake_claim]
        row[f"auc_{nm}"] = float(stats.mannwhitneyu(s_false, s_true).statistic / (n_te * n_te))
        row[f"acc_{nm}"] = float(np.mean(D.argmin(1) == Ck.test_lab))
    tn8.append(row)
    print(tn8[-1])
RESULTS["tn8"] = tn8

fig, ax = plt.subplots(figsize=(6.5, 3.4))
xk = [r["K"] for r in tn8]
ax.plot(xk, [r["auc_cos"] for r in tn8], "o-", color=C1, ms=6, label="AUC – Cosine Delta")
ax.plot(xk, [r["auc_classic"] for r in tn8], "o--", color=GRAY, ms=5, lw=1.5, label="AUC – Delta cổ điển")
ax.plot(xk, [r["acc_cos"] for r in tn8], "s-", color=C3, ms=5, label="Đoán đúng tác giả (50 lớp) – Cosine Delta")
ax.set_xscale("log"); ax.set_xticks(xk, [str(k) for k in xk]); ax.minorticks_off(); ax.set_xlabel("Số từ đặc trưng K (thang log)"); ax.set_ylabel("Giá trị")
ax.set_ylim(0, 1.02); ax.set_title("TN8 – Chọn số từ đặc trưng K"); ax.legend(fontsize=8)
savefig("08_tn8_so_tu_K.png")

# %% [markdown]
# ## Lưu kết quả

# %%
def _clean(o):
    if isinstance(o, dict): return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    return o
RESULTS["runtime_sec"] = time.time() - t0
with open(os.path.join(RES, "ket_qua.json"), "w", encoding="utf-8") as f:
    json.dump(_clean(RESULTS), f, ensure_ascii=False, indent=2)
print(f"Xong toàn bộ thí nghiệm trong {RESULTS['runtime_sec']/60:.1f} phút.")
