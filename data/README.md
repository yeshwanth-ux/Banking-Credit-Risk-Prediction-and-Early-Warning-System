# Data Directory

Raw and processed datasets are excluded from Git. Keep only documentation and
placeholder files in the repository.

Download and validate the selected Day 2 dataset with:

```powershell
.\.venv\Scripts\python.exe -m src.download_data
```

The command verifies the official UCI archive and source-file SHA-256 checksums,
then writes the official source materials and normalized `banking_data.csv` into
`data/raw/`.

Do not place confidential, personally identifiable, or unlicensed data in this
directory.
