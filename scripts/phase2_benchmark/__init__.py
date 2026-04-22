"""Phase 2 py-vs-R wallclock + regression-detection harness for pyinfercnv.

This is **not** a real-world validation benchmark. It measures
py-vs-R agreement (Jaccard on HMM state matrices; ARI on Leiden
subcluster partitions; wallclock ratio) on a fixed set of pycopykat-
sliced patient inputs. Both sides consume the same 3CA-derived
cell-type annotations as ``ref_group_names``; because 3CA annotations
are upstream automated labels, not expert-curated ground truth,
consistency between py and R here does not imply either side is
correct against external CNV truth. Use it for:

  (a) per-patient py-vs-R wallclock speedup measurement, and
  (b) regression detection — flagging algorithmic drift between
      pyinfercnv releases (e.g. 0.2.0.dev3 dropping DCIS1 i3 Jaccard
      from 0.887 to 0.5 would indicate a genuine regression).

Correctness-validated parity lives in ``tests/test_r_parity.py`` on
R infercnv's own built-in oligodendroglioma smart-seq2 fixture.

Reuses pycopykat's pre-sliced patient counts (benchmarks/full/<Cancer>/
<Patient>/{counts.tsv, cells.csv}) as read-only inputs; writes all outputs
under pyinfercnv/benchmarks/phase2/. Mirrors pycopykat/scripts/ layout:

    phase2_benchmark/
      dataset_manifest.py         patient registry + ref_group policy
      export_gene_order.py        parquet -> TSV for R infercnv
      prepare_annotations.py      cells.csv -> infercnv annotations_file
      run_r_phase2.R              R infercnv up_to_step=17 (per patient, per HMM_type)
      run_py_phase2.py            pyinfercnv Phase 2 (per patient, per HMM_type)
      compare_py_vs_r.py          per-patient metrics: i3/i6 Jaccard, step15 ARI
      run_all.py                  orchestrator (serial, resumable)
      aggregate.py                top-level phase2_py_vs_r_summary.csv
"""
