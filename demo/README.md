# Release demonstration

This directory contains synthetic, anonymized examples for the v1.0.0 project
demonstration. No row represents a real customer, account, or lending decision.

Run the demo after creating the local MLflow registry:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_release_demo
```

The demonstration performs four visible stages:

1. validates three CSV rows against the ingestion data contract;
2. validates three prediction requests through the FastAPI Pydantic schema;
3. loads the registered champion model and applies model bands plus rule-based
   early warnings;
4. exports an anonymized prediction JSON, portfolio summary, High Risk review
   queue, observed-period table, and readable HTML report.

The samples intentionally illustrate Low, Medium, and High outcomes. The
Medium example triggers missed-payment and worsening-delay rules. The High
example triggers severe delinquency, increasing debt burden, and previous-
default rules. These are demonstrations, not historical findings.

Generated release artifacts are committed under `demo/output/` so reviewers can
inspect the expected output without running local services. Regenerating them
replaces the files with equivalent results and a new generation timestamp.
