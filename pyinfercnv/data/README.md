# Bundled gene-position references

Each parquet has columns: `gene_symbol, chromosome, start, end`.

| File | Source | Generated via |
|---|---|---|
| `hg38_gene_positions.parquet` | GENCODE v45 basic annotation | `scripts/generate_gene_positions.py` |
| `hg19_gene_positions.parquet` | GENCODE v19 annotation | same |
| `mm10_gene_positions.parquet` | GENCODE vM25 annotation | same |

To regenerate (e.g. a new GENCODE release), download the corresponding `.gtf.gz`
from ftp.ebi.ac.uk/pub/databases/gencode/ and rerun the script:

```bash
python scripts/generate_gene_positions.py \
    --gtf /path/to/gencode.vXX.annotation.gtf.gz \
    --output pyinfercnv/data/<genome>_gene_positions.parquet \
    --genome <genome>
```
