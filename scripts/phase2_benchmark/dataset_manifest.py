"""17-patient benchmark manifest. Mirrors pycopykat/scripts/run_all_benchmarks.py
DATASETS list — keeps identical (cancer, sample) population so phase2 parity
numbers can be read alongside pycopykat's 17-patient copykat benchmark.

Reference-group policy for R infercnv (`ref_group_names`):
    observation = "Malignant"
    reference   = every non-empty cell_type != "Malignant"
    exclude     = cell_type == "" (unlabeled)

Annotation file format (R expects 2 tab-separated columns, no header):
    <cell_id>\\t<annotation>
where `annotation` is one of:
    malignant_<patient>          (all Malignant cells, tumor observation)
    <cell_type>                  (kept verbatim for reference groups)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Patient:
    cancer: str
    sample: str

    @property
    def pycopykat_dir(self) -> Path:
        """Read-only input dir under pycopykat/benchmarks/full/."""
        return PYCOPYKAT_ROOT / self.cancer / self.sample

    def out_dir(self, base: Path) -> Path:
        return base / self.cancer / self.sample


PYCOPYKAT_ROOT = Path("/media/jason/T7/rerbulid/pycopykat/benchmarks/full")
PYINFERCNV_ROOT = Path("/media/jason/T7/rerbulid/pyinfercnv")
BENCHMARK_OUT = PYINFERCNV_ROOT / "benchmarks" / "phase2"

# Identical to pycopykat's DATASETS tuple list (scripts/run_all_benchmarks.py).
PATIENTS: tuple[Patient, ...] = (
    Patient("Gao2021_Breast", "DCIS1"),
    Patient("Gao2021_Breast", "TNBC1"),
    Patient("Gao2021_Breast", "TNBC2"),
    Patient("Gao2021_Breast", "TNBC3"),
    Patient("Kim2020_Lung", "P1028"),
    Patient("Kim2020_Lung", "P0019"),
    Patient("Kim2020_Lung", "P0034"),
    Patient("Lee2020_Colorectal", "SMC16"),
    Patient("Lee2020_Colorectal", "SMC09"),
    Patient("Lee2020_Colorectal", "SMC21"),
    Patient("Obradovic2021_Kidney", "Patient4"),
    Patient("Obradovic2021_Kidney", "Patient5"),
    Patient("Obradovic2021_Kidney", "Patient2"),
    Patient("Qian2020_Ovarian", "11"),
    Patient("Qian2020_Ovarian", "14"),
    Patient("Qian2020_Ovarian", "12"),
    Patient("Qian2020_Ovarian", "13"),
)

HMM_TYPES: tuple[str, ...] = ("i6", "i3")  # both, per R r_reference.R


def lookup(cancer: str, sample: str) -> Patient:
    for p in PATIENTS:
        if p.cancer == cancer and p.sample == sample:
            return p
    raise KeyError(f"({cancer}, {sample}) not in manifest")
