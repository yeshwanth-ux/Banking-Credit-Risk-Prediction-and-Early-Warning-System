# Final architecture and workflow diagrams

## Runtime architecture

```mermaid
flowchart TB
    subgraph Sources[Authorized or public source zone]
        CSV[Raw CSV extracts]
        OPS[Customer, loan, repayment and account events]
    end

    subgraph Orchestration[Airflow orchestration]
        VALIDATE[Validate and quarantine]
        INGEST[Idempotent ingestion]
        CLEAN[Trusted point-in-time views]
        FEATURES[Feature engineering]
        SCORE[Batch scoring]
        REPORT[CSV and HTML reporting]
    end

    subgraph Data[Persistent services]
        PG[(PostgreSQL)]
        MLFLOW[(MLflow registry and artifacts)]
    end

    subgraph Serving[FastAPI service]
        AUTH[Role-aware authentication]
        PREDICT[Single and batch prediction]
        RISKAPI[Risk and portfolio APIs]
        MONITOR[Metrics, drift and scoring age]
    end

    subgraph Review[Human control]
        QUEUE[High Risk investigation queue]
        ANALYST[Authorized analyst review]
    end

    CSV --> VALIDATE
    OPS --> VALIDATE
    VALIDATE --> INGEST --> PG
    PG --> CLEAN --> FEATURES --> SCORE
    MLFLOW --> SCORE
    SCORE --> PG
    PG --> REPORT --> QUEUE --> ANALYST
    AUTH --> PREDICT
    AUTH --> RISKAPI
    AUTH --> MONITOR
    MLFLOW --> PREDICT
    PREDICT --> PG
    PG --> RISKAPI
    PREDICT --> MONITOR
    RISKAPI --> ANALYST
    MONITOR --> ANALYST
```

## Daily batch workflow

```mermaid
flowchart LR
    A[validate_source] --> B[ingest_source]
    B --> C[clean_and_validate]
    C --> D[generate_features]
    D --> E[batch_risk_prediction]
    E --> F[persist_predictions]
    F --> G[refresh_portfolio_reports]
```

Every logical run derives an idempotency key from its date and source hash.
Retries may update source records but cannot create a second scoring result set
for the same key.

## Online prediction and review sequence

```mermaid
sequenceDiagram
    actor Client as Authorized client
    participant API as FastAPI
    participant Registry as MLflow
    participant Model as Frozen pipeline
    participant DB as PostgreSQL
    participant Monitor as Monitoring
    actor Analyst as Risk analyst

    API->>Registry: Load champion and verify metadata at startup
    Client->>API: Submit schema-valid prediction
    API->>Model: Preprocess and predict probability
    API->>Model: Apply risk bands and warning rules
    API->>DB: Persist score, model version and warnings
    API->>Monitor: Aggregate latency, errors, risk mix and drift
    API-->>Client: Probability, band, explanations and warnings
    Analyst->>API: Authenticated risk/report request
    API->>DB: Query permitted latest records
    API-->>Analyst: Restricted review output
```

## Trust boundaries

- Secrets stay in environment configuration and are excluded from Git.
- Containers communicate on the Compose network; only configured host ports are
  exposed for local development.
- Raw inputs and feature snapshots are excluded from structured logs.
- Account-level reports require `risk_analyst` or `admin`; aggregate and
  monitoring access use separate roles.
- Model output supports manual review and never directly approves or declines a
  loan in this project.
