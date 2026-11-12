"""Prepare the benchmark tasks: GSE148673 breast tumours (counts + CopyKAT labels) and the gene order
(GENCODE v24, v32 for names v24 lacks). Run in the data directory containing raw/ and the GENCODE GTFs."""
import gzip, re, numpy as np, pandas as pd, scipy.sparse as sp, scipy.io, pathlib
D = pathlib.Path(".")
rows = []
for gtf in ["gencode.v24.annotation.gtf.gz", "gencode.v32.annotation.gtf.gz"]:   # v24 first, v32 for newer names
    for line in gzip.open(gtf, "rt"):
        if line.startswith("#"): continue
        f = line.split("\t", 9)
        if f[2] != "gene": continue
        rows.append((re.search(r'gene_name "([^"]+)"', f[8]).group(1), f[0], int(f[3]), int(f[4])))
go = pd.DataFrame(rows, columns=["gene", "chr", "start", "end"]).drop_duplicates("gene", keep="first")
go = go[go.chr.isin([f"chr{i}" for i in list(range(1, 23)) + ["X", "Y"]])]
go.to_csv("gene_order_gencode_v24.tsv", sep="\t", header=False, index=False)
for f in sorted((D / "raw").glob("GSM*.txt.gz")):
    s = f.name.split("_")[-1].split(".")[0]
    with gzip.open(f, "rt") as fh:
        cells = fh.readline().rstrip("\n").split("\t")
        pred = fh.readline().rstrip("\n").split("\t")[1:]
        fh.readline()
        df = pd.read_csv(fh, sep="\t", header=None, index_col=0)
    if len(cells) == df.shape[1] + 1: cells = cells[1:]
    assert len(cells) == df.shape[1] == len(pred), (len(cells), df.shape, len(pred))
    df = df[~df.index.duplicated()]
    keep = df.index.isin(go.gene)
    out = D / "tasks" / s; out.mkdir(parents=True, exist_ok=True)
    m = sp.csc_matrix(df[keep].to_numpy(dtype=np.float64))
    scipy.io.mmwrite(str(out / "counts.mtx"), m)
    pd.Series(df.index[keep]).to_csv(out / "genes.txt", index=False, header=False)
    pd.Series(cells).to_csv(out / "cells.txt", index=False, header=False)
    ann = pd.Series(np.where(np.asarray(pred) == "N", "normal", "tumor"), index=cells)
    ann.to_csv(out / "annotations.txt", sep="\t", header=False)
    print(s, m.shape, ann.value_counts().to_dict(), "genes kept", keep.sum(), "of", len(df))

# The oligodendroglioma example of inferCNV (argument: R's system.file("extdata", package = "infercnv"))
import shutil, sys
if len(sys.argv) > 1:
    ext = pathlib.Path(sys.argv[1])
    out = D / "tasks" / "oligodendroglioma"; out.mkdir(parents=True, exist_ok=True)
    cnt = pd.read_csv(ext / "oligodendroglioma_expression_downsampled.counts.matrix.gz", sep="\t", index_col=0)
    scipy.io.mmwrite(str(out / "counts.mtx"), sp.csc_matrix(cnt.to_numpy(dtype=np.float64)), precision=17)
    pd.Series(cnt.index).to_csv(out / "genes.txt", index=False, header=False)
    pd.Series(cnt.columns).to_csv(out / "cells.txt", index=False, header=False)
    ann = pd.read_csv(ext / "oligodendroglioma_annotations_downsampled.txt", sep="\t", header=None, index_col=0)[1]
    ann.reindex(cnt.columns).to_csv(out / "annotations.txt", sep="\t", header=False)
    (out / "ref_groups.txt").write_text("Microglia/Macrophage\nOligodendrocytes (non-malignant)\n")
    (out / "cutoff.txt").write_text("1\n")
    shutil.copy(ext / "gencode_downsampled.EXAMPLE_ONLY_DONT_REUSE.txt", out / "gene_order.tsv")
    print("oligodendroglioma", cnt.shape)
