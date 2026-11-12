"""Figure and the numbers quoted in the pyinfercnv technical report.

Usage: python 05_figures.py <runs_dir> <tasks_dir> <out_dir>
Compares R inferCNV 1.26.0 and pyinfercnv on the oligodendroglioma example of inferCNV (Smart-seq2) and
six breast tumours from 10x Genomics (GSE148673), all run with HMM (i6), subclusters and denoising from
set.seed(1) / random_state=1.
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy
import scipy.io
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score

RUNS, TASKS, OUT = map(pathlib.Path, sys.argv[1:4])
OUT.mkdir(parents=True, exist_ok=True)
ASSETS = pathlib.Path(__file__).resolve().parent.parent / "assets"
C_R, C_PY = "#eb6834", "#2a78d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 300,
})
ENG = {"R": "infercnv", "py": "pyinfercnv"}
LABEL = {"oligodendroglioma": "Oligodendroglioma", "DCIS1": "DCIS1", "TNBC1": "TNBC1", "TNBC2": "TNBC2",
         "TNBC3": "TNBC3", "TNBC4": "TNBC4", "TNBC5": "TNBC5"}
EXAMPLE = "TNBC1"


def load(engine, task, tag, run=1):
    d = RUNS / ENG[engine] / task / f"run{run}"
    g = [x for x in (d / f"{tag}_genes.txt").read_text().split("\n") if x]
    c = (d / f"{tag}_cells.txt").read_text().split()
    return pd.DataFrame(np.fromfile(d / f"{tag}.bin").reshape(len(c), len(g)).T, index=g, columns=c)


def seconds(engine, task):
    return min(json.load(open(RUNS / ENG[engine] / task / f"run{k}" / "result.json"))["seconds"] for k in (1, 2))


rows, keep = [], {}
for t in LABEL:
    ann = pd.read_csv(TASKS / t / "annotations.txt", sep="\t", header=None, index_col=0)[1]
    refs = ((TASKS / t / "ref_groups.txt").read_text().splitlines() if (TASKS / t / "ref_groups.txt").exists()
            else ["normal"])
    row = dict(dataset=t, label=LABEL[t])
    mats = {}
    for tag in ["14", "17", "22"]:
        r, p = load("R", t, tag), load("py", t, tag)
        row[f"same_genes_{tag}"] = list(r.index) == list(p.index)
        p = p.loc[r.index, r.columns]
        d = np.abs(p.values - r.values)
        row[f"maxdiff_{tag}"] = float(d.max())
        row[f"agree_{tag}"] = float((d < 1e-6).mean())
        mats[tag] = (r, p)
    row["genes"], row["cells"] = mats["14"][0].shape
    row["n_ref"] = int(ann.isin(refs).sum())
    # subclusters: ARI within each group, pooled per tumour/normal side
    sr = pd.read_csv(RUNS / ENG["R"] / t / "run1" / "subclusters.tsv", sep="\t").set_index("cell")
    spy = pd.read_csv(RUNS / ENG["py"] / t / "run1" / "subclusters.tsv", sep="\t").set_index("cell")
    aris, nR, nP = [], 0, 0
    for g in sr.group.unique():
        c = sr.index[sr.group == g]
        aris.append((len(c), adjusted_rand_score(sr.loc[c, "subcluster"], spy.loc[c, "subcluster"])))
        nR += sr.loc[c, "subcluster"].nunique()
        nP += spy.loc[c, "subcluster"].nunique()
    row["ari"] = float(sum(n * a for n, a in aris) / sum(n for n, _ in aris))   # cell-weighted mean over groups
    row["nsub_R"], row["nsub_py"] = nR, nP
    row["identical_sub"] = all(abs(a - 1) < 1e-12 for _, a in aris)
    row["sec_R"], row["sec_py"] = seconds("R", t), seconds("py", t)
    keep[t] = (mats, ann, refs)
    rows.append(row)
tab = pd.DataFrame(rows)
tab.to_csv(OUT / "comparison_per_dataset.tsv", sep="\t", index=False)

# ---- oligodendroglioma: 1p / 19q in the malignant cells (genes ordered as in the gene-order file) ----
go = pd.read_csv(TASKS / "oligodendroglioma" / "gene_order.tsv", sep="\t", header=None, index_col=0,
                 names=["g", "chr", "start", "end"])
mats, ann, refs = keep["oligodendroglioma"]
r14, p14 = mats["14"]
mal = [c for c in r14.columns if not ann.get(c, "") in refs]
pos = go.reindex(r14.index)
arm = np.where((pos.chr == "chr1") & (pos.start < 121_107_124), "1p",
               np.where((pos.chr == "chr19") & (pos.start > 24_216_276), "19q", "other"))
oligo = {}
for nm, M in [("R", r14), ("py", p14)]:
    v = M[mal]
    oligo[nm] = {a: float(v[arm == a].to_numpy().mean()) for a in ["1p", "19q", "other"]}
    oligo[nm]["genome"] = float(v.to_numpy().mean())

# ---- UMAP of the example (display only) and CNV score per cell ----
td = TASKS / EXAMPLE
X = sp.csr_matrix(scipy.io.mmread(td / "counts.mtx").T)
cells = (td / "cells.txt").read_text().split()
a = scanpy.AnnData(X.astype(np.float32))
scanpy.pp.normalize_total(a, target_sum=1e4)
scanpy.pp.log1p(a)
scanpy.pp.highly_variable_genes(a, n_top_genes=2000, subset=True)
scanpy.pp.pca(a, n_comps=30, random_state=0)
scanpy.pp.neighbors(a, n_neighbors=15, random_state=0)
scanpy.tl.umap(a, random_state=0)
xy = a.obsm["X_umap"]
exm, exann, exrefs = keep[EXAMPLE]
score = {nm: ((exm["22"][i] - 1.0) ** 2).mean(axis=0).reindex(cells).to_numpy() for i, nm in [(0, "R"), (1, "py")]}
hmm_alt = {nm: (exm["17"][i] != 3).mean(axis=0).reindex(cells).to_numpy() for i, nm in [(0, "R"), (1, "py")]}

# ======================= Figure 1 =======================
fig = plt.figure(figsize=(7.2, 6.6))
gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.05, 0.95], hspace=0.6)


def panel(ax, s, dx=-0.36):
    ax.text(dx, 1.02, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left", color=INK)


def crop(img):
    ink = (img[..., 3] > 0.04) & (img[..., :3].min(-1) < 0.96)
    ys, xs = np.where(ink)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


top = gs[0].subgridspec(1, 4, width_ratios=[0.62, 1, 1, 1], wspace=0.6)
logos = [crop(plt.imread(ASSETS / f)) for f in ["pyinfercnv_logo.png", "omicverse_logo.png"]]
x = np.arange(len(tab))

axa = fig.add_subplot(top[1])
pts = []
for i, t in enumerate(tab.dataset):
    r, p = keep[t][0]["22"]
    rv, pv = r.to_numpy().ravel(), p.to_numpy().ravel()
    pick = np.random.default_rng(i).choice(rv.size, min(4000, rv.size), replace=False)
    pts.append((rv[pick], pv[pick]))
rv = np.concatenate([q[0] for q in pts]); pv = np.concatenate([q[1] for q in pts])
lo, hi = min(rv.min(), pv.min()), max(rv.max(), pv.max())
axa.plot([lo, hi], [lo, hi], color=GRID, lw=0.6, zorder=0)
axa.scatter(rv, pv, s=0.6, color=C_PY, lw=0, rasterized=True)
axa.set_xlabel("inferCNV (R)")
axa.set_ylabel("pyinfercnv")
axa.set_title("Denoised CNV matrix")
axa.set_aspect("equal")
axa.text(0.96, 0.04, f"7 datasets\nmax |R - Python|\n{tab.maxdiff_22.max():.0e}", transform=axa.transAxes,
         ha="right", va="bottom", fontsize=5, color=INK2)
panel(axa, "a")

axb = fig.add_subplot(top[2])
axb.bar(x - 0.2, tab.agree_17 * 100, 0.38, color="#2f7f6f", label="HMM states")
axb.bar(x + 0.2, tab.ari * 100, 0.38, color="#b5a642", label="subclusters (ARI)")
axb.set_xticks(x, tab.label, rotation=40, ha="right", rotation_mode="anchor", fontsize=5.5)
axb.set_ylim(70, 102)
axb.set_ylabel("agreement with R (%)")
axb.set_title("HMM and subclusters", pad=13)
axb.grid(axis="y", color=GRID, lw=0.5)
axb.set_axisbelow(True)
axb.legend(fontsize=5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, handlelength=1, handletextpad=0.3,
           columnspacing=0.6, borderaxespad=0.1)
panel(axb, "b")

axc = fig.add_subplot(top[3])
axc.scatter(tab.cells, tab.sec_R, s=12, color=C_R, edgecolor="white", linewidth=0.3, label="inferCNV 1.26 (R)")
axc.scatter(tab.cells, tab.sec_py, s=12, color=C_PY, edgecolor="white", linewidth=0.3, label="pyinfercnv")
for _, r in tab.iterrows():
    axc.plot([r.cells] * 2, [r.sec_py, r.sec_R], color=GRID, lw=0.8, zorder=0)
axc.set_xscale("log")
axc.set_yscale("log")
axc.set_xticks([200, 500, 1000, 3000], ["200", "500", "1,000", "3,000"])
axc.set_yticks([3, 10, 30, 100, 300, 1000], ["3", "10", "30", "100", "300", "1000"])
axc.minorticks_off()
axc.set_xlabel("cells")
axc.set_ylabel("wall time, 1 thread (s)")
axc.set_title("Run time")
axc.grid(color=GRID, lw=0.5)
axc.set_axisbelow(True)
axc.legend(fontsize=5, loc="center right", handletextpad=0.1)
panel(axc, "c")

# heatmaps: oligodendroglioma malignant cells, R and Python, genes in genomic order
mid = gs[1].subgridspec(1, 2, width_ratios=[1.0, 0.4], wspace=0.32)
hm = mid[0].subgridspec(2, 1, hspace=0.12)
chroms = pos.chr.to_numpy()
bounds = np.flatnonzero(chroms[1:] != chroms[:-1]) + 1
order = sorted(mal, key=lambda c: (ann[c], c))
r22, p22 = mats["22"]
for k, (nm, M) in enumerate([("inferCNV (R)", r22), ("pyinfercnv", p22)]):
    ax = fig.add_subplot(hm[k])
    im = ax.imshow(M[order].to_numpy().T, aspect="auto", cmap="RdBu_r", vmin=0.85, vmax=1.15,
                   interpolation="nearest", rasterized=True)
    for b in bounds:
        ax.axvline(b - 0.5, color=INK2, lw=0.3)
    ax.set_yticks([])
    ax.set_ylabel(nm, fontsize=6)
    if k == 1:
        centers = np.concatenate(([0], bounds, [len(chroms)]))
        mids = (centers[:-1] + centers[1:]) / 2
        names = [c.replace("chr", "") for c in pd.unique(chroms)]
        ax.set_xticks(mids, names, fontsize=4.8)
        ax.tick_params(axis="x", length=0)
        ax.set_xlabel("chromosome")
    else:
        ax.set_xticks([])
        ax.set_title(f"Oligodendroglioma: denoised CNV of {len(mal)} malignant cells")
        panel(ax, "d", dx=-0.07)
cax = fig.add_axes([0, 0, 1, 1])
cax.set_visible(False)

axe = fig.add_subplot(mid[1])
arms = ["1p", "19q", "other"]
xa = np.arange(3)
axe.bar(xa - 0.2, [oligo["R"][a] - 1 for a in arms], 0.38, color=C_R, label="inferCNV (R)")
axe.bar(xa + 0.2, [oligo["py"][a] - 1 for a in arms], 0.38, color=C_PY, label="pyinfercnv")
axe.axhline(0, color=INK2, lw=0.5)
axe.set_xticks(xa, ["1p", "19q", "rest of\ngenome"])
axe.set_ylabel("mean relative expression - 1")
axe.set_title("1p/19q co-deletion")
axe.grid(axis="y", color=GRID, lw=0.5)
axe.set_axisbelow(True)
axe.legend(fontsize=5, loc="lower right", handletextpad=0.3)
panel(axe, "e", dx=-0.45)

row3 = gs[2].subgridspec(1, 4, wspace=0.06)


def umap_ax(k, title, letter):
    ax = fig.add_subplot(row3[k])
    ax.set_xticks([])
    ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_title(title)
    ax.set_aspect("equal")
    panel(ax, letter, dx=-0.04)
    return ax


ax = umap_ax(0, f"{EXAMPLE}: CopyKAT label", "f")
lab = exann.reindex(cells).to_numpy()
for name, col in [("normal", "#2f7f6f"), ("tumor", "#b5a642")]:
    m = lab == name
    ax.scatter(xy[m, 0], xy[m, 1], s=0.8, c=col, lw=0, rasterized=True, label=f"{name} ({m.sum():,})")
leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=5.5, handletextpad=0.1)
for hh in leg.legend_handles:
    hh.set_sizes([10])
vmax = np.nanpercentile(np.concatenate([score["R"], score["py"]]), 99)
for k, (nm, title, letter) in enumerate([("R", "CNV score, inferCNV (R)", "g"), ("py", "CNV score, pyinfercnv", "h")]):
    ax = umap_ax(k + 1, title, letter)
    o = np.argsort(score[nm], kind="stable")
    im = ax.scatter(xy[o, 0], xy[o, 1], c=score[nm][o], s=0.8, cmap="magma_r", vmin=0, vmax=vmax, lw=0,
                    rasterized=True)
cb_ax = fig.axes[-1].inset_axes([-0.55, -0.12, 1.1, 0.035])
cb = fig.colorbar(im, cax=cb_ax, orientation="horizontal")
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=5.5, length=1.5)
cb_ax.set_title("mean squared deviation of the denoised matrix from 1", fontsize=5.5, color=INK2, pad=2)
ax = umap_ax(3, "HMM: genes with a CNV", "i")
diff = hmm_alt["py"] - hmm_alt["R"]
o = np.argsort(hmm_alt["py"], kind="stable")
im2 = ax.scatter(xy[o, 0], xy[o, 1], c=hmm_alt["py"][o] * 100, s=0.8, cmap="viridis", lw=0, rasterized=True)
cb2_ax = ax.inset_axes([0.05, -0.12, 0.9, 0.035])
cb2 = fig.colorbar(im2, cax=cb2_ax, orientation="horizontal")
cb2.outline.set_visible(False)
cb2.ax.tick_params(labelsize=5.5, length=1.5)
cb2_ax.set_title("% of genes, pyinfercnv", fontsize=5.5, color=INK2, pad=2)

pa, pd_ = axa.get_position(), fig.axes[4].get_position()
x0, x1 = 0.02, pa.x0 - 0.075
y0, y1 = pa.y0 - 0.02, pa.y1 + 0.035
h_ov = (x1 - x0) * 7.2 / 6.6 * logos[1].shape[0] / logos[1].shape[1]
for img, rect, anchor in [(logos[0], [x0, y0 + h_ov + 0.012, x1 - x0, y1 - y0 - h_ov - 0.012], "N"),
                          (logos[1], [x0, y0, x1 - x0, h_ov], "S")]:
    axl = fig.add_axes(rect)
    axl.imshow(img)
    axl.set_anchor(anchor)
    axl.set_axis_off()
fig.savefig(OUT / "fig1.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "fig1.png", bbox_inches="tight", pad_inches=0.02, dpi=200)

# ======================= numbers =======================
tenx = tab[tab.dataset != "oligodendroglioma"]
ol = tab.set_index("dataset").loc["oligodendroglioma"]
speed = tab.sec_R / tab.sec_py
big = tab.iloc[tab.cells.idxmax()]
macros = {
    "nDatasets": len(tab), "nTenx": len(tenx), "nCellsTenxTotal": f"{int(tenx.cells.sum()):,}",
    "nCellsMin": f"{int(tenx.cells.min()):,}", "nCellsMax": f"{int(tenx.cells.max()):,}",
    "oligoCells": int(ol.cells), "oligoGenes": f"{int(ol.genes):,}", "oligoMal": len(mal),
    "maxDiffExpr": f"{max(tab.maxdiff_14.max(), tab.maxdiff_22.max()):.0e}",
    "agreeMin": f"{tenx.agree_17.min() * 100:.1f}", "agreeMax": f"{tenx.agree_17.max() * 100:.1f}",
    "ariMin": f"{tenx.ari.min():.2f}", "ariMax": f"{tenx.ari.max():.2f}",
    "speedMin": f"{speed.min():.0f}", "speedMax": f"{speed.max():.0f}",
    "tBigR": f"{big.sec_R:.0f}", "tBigPy": f"{big.sec_py:.0f}", "bigName": big.label,
    "oligoR": f"{ol.sec_R:.0f}", "oligoPy": f"{ol.sec_py:.1f}",
    "oneP": f"{(oligo['R']['1p'] - 1) * 100:.1f}", "nineteenQ": f"{(oligo['R']['19q'] - 1) * 100:.1f}",
    "restGenome": f"{(oligo['R']['other'] - 1) * 100:.1f}",
}
with open(OUT / "numbers.tex", "w") as fh:
    for k, v in macros.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(open(OUT / "numbers.tex").read())
print(tab.drop(columns=["label"]).round(6).T.to_string())
print(oligo)
