# South German Credit Data Dictionary

All source fields are integer-coded. Categorical and ordinal codes must be
decoded or explicitly modeled as categories; they are not continuous quantities.
The official UCI code table in `data/raw/codetable.txt` is the coding authority.

| Field | Source | Role | Group | Modeling type | Definition and units | Expected missing | Codes or range | Risk note |
|---|---|---|---|---|---|---:|---|---|
| `checking_account_status` | `laufkont` | Feature | Credit indicator | Categorical | Status of checking account | 0 | 1 no account; 2 below 0 DM; 3 0-199 DM; 4 at least 200 DM or salary assignment | Confirm value is known at prediction time |
| `duration_months` | `laufzeit` | Feature | Loan detail | Numeric | Credit duration in months | 0 | Positive integer | None identified |
| `credit_history` | `moral` | Feature | Repayment indicator | Categorical | Compliance history for previous or concurrent credit contracts | 0 | 0-4; see official code table | Temporal leakage risk because concurrent-contract timing is unavailable |
| `purpose` | `verw` | Feature | Loan detail | Categorical | Purpose for requested credit | 0 | 0-10; see official code table | None identified |
| `credit_amount_dm` | `hoehe` | Feature | Loan detail | Numeric | Credit amount in DM after an undocumented monotonic transformation | 0 | Positive integer | Do not interpret as original currency amount |
| `savings` | `sparkont` | Feature | Credit indicator | Ordinal categorical | Savings-account band | 0 | 1 unknown/no account; 2 below 100 DM; 3 100-499; 4 500-999; 5 at least 1000 | Coarse financial proxy |
| `employment_duration` | `beszeit` | Feature | Customer | Ordinal categorical | Time with current employer | 0 | 1 unemployed; 2 below 1 year; 3 1-3 years; 4 4-6 years; 5 at least 7 years | Employment proxy may affect fairness |
| `installment_rate` | `rate` | Feature | Affordability | Ordinal categorical | Installment as a percentage of disposable income | 0 | 1 at least 35%; 2 25-34%; 3 20-24%; 4 below 20% | Reverse-coded burden bands |
| `personal_status_sex` | `famges` | Feature | Customer | Categorical | Combined sex and marital-status code | 0 | 1-4; see official code table | Sensitive; categories do not permit clean sex recovery |
| `other_debtors` | `buerge` | Feature | Loan detail | Categorical | Other debtor or guarantor | 0 | 1 none; 2 co-applicant; 3 guarantor | None identified |
| `present_residence` | `wohnzeit` | Feature | Customer | Ordinal categorical | Years at present residence | 0 | 1 below 1 year; 2 1-3; 3 4-6; 4 at least 7 | Stability proxy |
| `property` | `verm` | Feature | Credit indicator | Ordinal categorical | Most valuable property category | 0 | 1 unknown/none; 2 car/other; 3 savings agreement/life insurance; 4 real estate | Wealth proxy |
| `age_years` | `alter` | Feature | Customer | Numeric | Age in years | 0 | Positive integer | Sensitive/fairness-relevant |
| `other_installment_plans` | `weitkred` | Feature | Loan detail | Categorical | Installment plans outside the lending bank | 0 | 1 bank; 2 stores; 3 none | Confirm value is known at prediction time |
| `housing` | `wohn` | Feature | Customer | Categorical | Housing arrangement | 0 | 1 free; 2 rent; 3 own | Wealth and stability proxy |
| `number_credits` | `bishkred` | Feature | Credit indicator | Ordinal categorical | Number of current or prior credits at this bank | 0 | 1 one; 2 two-three; 3 four-five; 4 at least six | Confirm observation timing |
| `job` | `beruf` | Feature | Customer | Ordinal categorical | Job-skill and employment category | 0 | 1-4; see official code table | Employment and socioeconomic proxy |
| `people_liable` | `pers` | Feature | Customer | Binary categorical | People financially dependent on debtor | 0 | 1 three or more; 2 zero-two | Reverse-coded count band |
| `telephone` | `telef` | Feature | Customer | Binary categorical | Landline registered in customer's name | 0 | 1 no; 2 yes | Historically specific proxy from the 1970s |
| `foreign_worker` | `gastarb` | Feature | Customer | Binary categorical | Foreign-worker indicator | 0 | 1 yes; 2 no | Sensitive/fairness-relevant nationality proxy |
| `credit_risk` | `kredit` | Source target | Outcome | Binary categorical | Contract compliance outcome | 0 | 0 bad; 1 good | Leakage if included as a feature; inverse of `bad_credit` |
| `bad_credit` | Derived | Modeling target | Outcome | Binary categorical | Positive-class label for failed contract compliance | 0 | 0 good; 1 bad | Target only; not a 90-day default label |
