"""
mophong.py — Thư viện lõi cho đồ án
"Phát hiện bài báo bị ghi sai tác giả bằng NLP"

Ý tưởng chính
-------------
* Mỗi bài báo được biểu diễn bằng tần suất tương đối của K từ phổ biến nhất
  (Most Frequent Words – MFW), chủ yếu là hư từ (the, of, and, to...).
  Đây là "dấu vân tay văn phong" ít phụ thuộc chủ đề.
* Khoảng cách Burrows' Delta (bản cổ điển) giữa bài d và hồ sơ (centroid) tác giả A:
      Delta(d, A) = (1/K) * sum_k | z_k(d) - z_k(A) |
  với z là tần suất đã chuẩn hoá theo trung bình / độ lệch chuẩn của toàn kho.
  Thước đo chính dùng trong đồ án là Cosine Delta = 1 - cos(z_d, z_A),
  Delta cổ điển được giữ lại làm mốc so sánh.
* Kiểm định  H0: "bài d đúng là của tác giả A được ghi".
  Phân phối của Delta dưới H0 không có dạng giải tích, nên ta MÔ PHỎNG nó
  bằng bootstrap: lặp B lần
      - rút ngẫu nhiên 1 bài thật của A, cắt 1 đoạn ngẫu nhiên dài L từ
        (L = độ dài bài cần kiểm tra -> null "khớp độ dài"),
      - dựng centroid từ một mẫu bootstrap (lấy có hoàn lại) các bài còn lại của A,
      - tính khoảng cách.
  p-value Monte Carlo = (1 + #{Delta_b >= Delta_obs}) / (B + 1).
"""
from __future__ import annotations

import os
import re
import sys
import glob
from collections import Counter
from dataclasses import dataclass

import numpy as np

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


# ----------------------------------------------------------------------------
# 1. Đọc dữ liệu
# ----------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    # sys.intern: các từ giống nhau dùng chung một đối tượng -> tiết kiệm RAM
    # (quan trọng khi chạy web trên máy chủ ít bộ nhớ như Render free 512 MB)
    return [sys.intern(t) for t in TOKEN_RE.findall(text.lower())]


def load_c50(root: str):
    """Đọc bộ Reuter_50_50. root chứa C50train/ và C50test/.
    Trả về dict split -> list[(author, tokens)]."""
    out = {}
    for split in ("C50train", "C50test"):
        docs = []
        for adir in sorted(glob.glob(os.path.join(root, split, "*"))):
            if not os.path.isdir(adir):
                continue
            author = os.path.basename(adir)
            for f in sorted(glob.glob(os.path.join(adir, "*.txt"))):
                with open(f, encoding="utf-8", errors="ignore") as fh:
                    docs.append((author, tokenize(fh.read())))
        out[split] = docs
    return out


# ----------------------------------------------------------------------------
# 2. Biểu diễn văn phong
# ----------------------------------------------------------------------------
@dataclass
class Corpus:
    authors: list            # tên tác giả (index -> tên)
    vocab: list              # K từ MFW
    train_ids: list          # list[np.ndarray] token-id (−1 = ngoài MFW)
    train_lab: np.ndarray    # nhãn tác giả của bài train
    test_ids: list
    test_lab: np.ndarray
    mu: np.ndarray           # trung bình tần suất tương đối (K,)
    sd: np.ndarray           # độ lệch chuẩn (K,)

    @property
    def K(self):
        return len(self.vocab)


def build_corpus(data, K: int = 300) -> Corpus:
    train, test = data["C50train"], data["C50test"]
    authors = sorted({a for a, _ in train})
    a2i = {a: i for i, a in enumerate(authors)}
    cnt = Counter()
    for _, toks in train:
        cnt.update(toks)
    vocab = [w for w, _ in cnt.most_common(K)]
    w2i = {w: i for i, w in enumerate(vocab)}

    def enc(toks):
        return np.array([w2i.get(t, -1) for t in toks], dtype=np.int32)

    tr_ids = [enc(t) for _, t in train]
    te_ids = [enc(t) for _, t in test]
    tr_lab = np.array([a2i[a] for a, _ in train])
    te_lab = np.array([a2i[a] for a, _ in test])
    F = np.vstack([rel_freq(x, K) for x in tr_ids])
    mu, sd = F.mean(0), F.std(0) + 1e-12
    return Corpus(authors, vocab, tr_ids, tr_lab, te_ids, te_lab, mu, sd)


def rel_freq(ids: np.ndarray, K: int) -> np.ndarray:
    """Tần suất tương đối của K từ MFW trong một chuỗi token-id."""
    n = max(len(ids), 1)
    v = ids[ids >= 0]
    return np.bincount(v, minlength=K).astype(float) / n


def zscore(C: Corpus, F: np.ndarray) -> np.ndarray:
    return (F - C.mu) / C.sd


def random_chunk(ids: np.ndarray, L: int | None, rng) -> np.ndarray:
    """Cắt ngẫu nhiên một đoạn liên tục dài L token (None = cả bài)."""
    if L is None or len(ids) <= L:
        return ids
    s = rng.integers(0, len(ids) - L + 1)
    return ids[s:s + L]


def delta(zd: np.ndarray, zc: np.ndarray) -> np.ndarray:
    """Burrows' Delta cổ điển (hỗ trợ broadcasting)."""
    return np.abs(zd - zc).mean(-1)


def cosine_delta(zd: np.ndarray, zc: np.ndarray) -> np.ndarray:
    """Cosine Delta (Smith & Aldridge, 2011): 1 - cos(z_d, z_A).
    Thường ổn định hơn Delta cổ điển; là thước đo chính của đồ án."""
    num = (zd * zc).sum(-1)
    den = np.linalg.norm(zd, axis=-1) * np.linalg.norm(zc, axis=-1) + 1e-12
    return 1.0 - num / den


dist = cosine_delta   # thước đo mặc định


# ----------------------------------------------------------------------------
# 3. Kiểm định bootstrap / Monte Carlo
# ----------------------------------------------------------------------------
class AuthorProfile:
    """Hồ sơ văn phong của một tác giả + phân phối null mô phỏng."""

    def __init__(self, C: Corpus, doc_ids: list, rng, Zfull=None, metric=None):
        self.C = C
        self.metric = metric or dist
        self.docs = doc_ids                      # list[np.ndarray]
        self.Z = np.vstack([zscore(C, rel_freq(d, C.K)) for d in doc_ids]) \
            if Zfull is None else Zfull
        self.centroid = self.Z.mean(0)
        self.rng = rng
        self._null_cache = {}

    def null(self, L: int | None, B: int = 500) -> np.ndarray:
        """Mô phỏng B giá trị Delta dưới H0 cho bài dài L từ."""
        key = (L, B)
        if key in self._null_cache:
            return self._null_cache[key]
        n, rng, C = len(self.docs), self.rng, self.C
        out = np.empty(B)
        for b in range(B):
            j = rng.integers(n)                                   # bài bị "giữ lại"
            others = np.delete(np.arange(n), j)
            boot = rng.choice(others, size=n - 1, replace=True)   # bootstrap
            zc = self.Z[boot].mean(0)
            chunk = random_chunk(self.docs[j], L, rng)
            zd = zscore(C, rel_freq(chunk, C.K))
            out[b] = self.metric(zd, zc)
        self._null_cache[key] = out
        return out

    def stat(self, ids: np.ndarray) -> float:
        return float(self.metric(zscore(self.C, rel_freq(ids, self.C.K)), self.centroid))

    def pvalue(self, ids: np.ndarray, L: int | None = None, B: int = 500) -> float:
        T = self.stat(ids)
        nl = self.null(L, B)
        return (1 + np.sum(nl >= T)) / (B + 1)


def mc_pvalue(T: np.ndarray, null: np.ndarray) -> np.ndarray:
    """p-value Monte Carlo cho một mảng thống kê T với cùng một null."""
    null = np.sort(null)
    ge = len(null) - np.searchsorted(null, T, side="left")
    return (1 + ge) / (len(null) + 1)


def benjamini_hochberg(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Trả về mảng bool: bài nào bị đánh dấu nghi ngờ khi kiểm soát FDR = q."""
    m = len(p)
    order = np.argsort(p)
    thr = q * np.arange(1, m + 1) / m
    ok = p[order] <= thr
    rej = np.zeros(m, bool)
    if ok.any():
        k = np.max(np.where(ok)[0])
        rej[order[:k + 1]] = True
    return rej


# ----------------------------------------------------------------------------
# 4. Sinh văn bản giả bằng chuỗi Markov (bigram)
# ----------------------------------------------------------------------------
class BigramMarkov:
    """Chuỗi Markov bậc 1 trên từ, học từ các bài của một tác giả."""

    def __init__(self, token_lists: list[list[str]]):
        trans = {}
        for toks in token_lists:
            for a, b in zip(toks[:-1], toks[1:]):
                trans.setdefault(a, Counter())[b] += 1
        self.states = list(trans)
        self.next = {a: (list(c.keys()), np.array(list(c.values()), float))
                     for a, c in trans.items()}
        start = Counter(t[0] for t in token_lists if t)
        self.start = (list(start.keys()), np.array(list(start.values()), float))

    def sample(self, L: int, rng) -> list[str]:
        w, p = self.start
        cur = w[rng.choice(len(w), p=p / p.sum())]
        out = [cur]
        while len(out) < L:
            if cur not in self.next:
                w, p = self.start
            else:
                w, p = self.next[cur]
            cur = w[rng.choice(len(w), p=p / p.sum())]
            out.append(cur)
        return out
