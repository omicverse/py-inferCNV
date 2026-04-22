"""Phase 2 R-vs-Python benchmark harness for pyinfercnv.

Reuses pycopykat's pre-sliced 3CA patient counts (benchmarks/full/<Cancer>/
<Patient>/{counts.tsv, cells.csv}) as read-only inputs; writes all outputs
under pyinfercnv/benchmarks/phase2/. Mirrors pycopykat/scripts/ layout:

    phase2_benchmark/
      dataset_manifest.py         17-patient registry + ref_group policy
      export_gene_order.py        parquet -> TSV for R infercnv
      prepare_annotations.py      cells.csv -> infercnv annotations_file
      run_r_phase2.R              R infercnv up_to_step=17 (per patient, per HMM_type)
      run_py_phase2.py            pyinfercnv Phase 2 (per patient, per HMM_type)
      compare_py_vs_r.py          per-patient metrics: i3/i6 Jaccard, step15 ARI
      run_all.py                  orchestrator (serial, resumable)
      aggregate.py                top-level phase2_py_vs_r_summary.csv
"""
