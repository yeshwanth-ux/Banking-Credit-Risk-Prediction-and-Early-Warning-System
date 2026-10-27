CREATE TABLE scoring_runs (
    scoring_run_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_reference VARCHAR(64) NOT NULL UNIQUE,
    request_type VARCHAR(16) NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ NOT NULL,
    status VARCHAR(16) NOT NULL,
    model_name VARCHAR(128) NOT NULL,
    model_version VARCHAR(64) NOT NULL,
    model_uri VARCHAR(255) NOT NULL,
    configuration_version VARCHAR(64) NOT NULL,
    requested_by VARCHAR(64) NOT NULL,
    requested_count INTEGER NOT NULL,
    succeeded_count INTEGER NOT NULL,
    failed_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT scoring_runs_request_type_allowed
        CHECK (request_type IN ('single', 'batch')),
    CONSTRAINT scoring_runs_status_allowed
        CHECK (status IN ('completed', 'partial', 'failed')),
    CONSTRAINT scoring_runs_time_order
        CHECK (completed_at >= started_at),
    CONSTRAINT scoring_runs_counts_nonnegative
        CHECK (
            requested_count >= 0 AND succeeded_count >= 0 AND failed_count >= 0
        ),
    CONSTRAINT scoring_runs_counts_balance
        CHECK (requested_count = succeeded_count + failed_count)
);

ALTER TABLE risk_predictions
    ADD COLUMN scoring_run_id BIGINT,
    ADD COLUMN reference_identifier VARCHAR(128),
    ADD COLUMN model_uri VARCHAR(255),
    ADD COLUMN configuration_version VARCHAR(64);

UPDATE risk_predictions AS prediction
SET
    reference_identifier = loan.loan_reference,
    model_uri = prediction.model_name || ':' || prediction.model_version,
    configuration_version = 'legacy'
FROM loans AS loan
WHERE loan.loan_id = prediction.loan_id;

ALTER TABLE risk_predictions
    ALTER COLUMN loan_id DROP NOT NULL,
    ALTER COLUMN reference_identifier SET NOT NULL,
    ALTER COLUMN model_uri SET NOT NULL,
    ALTER COLUMN configuration_version SET NOT NULL,
    ADD CONSTRAINT risk_predictions_scoring_run_fk
        FOREIGN KEY (scoring_run_id) REFERENCES scoring_runs(scoring_run_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    ADD CONSTRAINT risk_predictions_reference_nonempty
        CHECK (length(trim(reference_identifier)) > 0),
    ADD CONSTRAINT risk_predictions_subject_required
        CHECK (loan_id IS NOT NULL OR reference_identifier IS NOT NULL);

CREATE UNIQUE INDEX idx_risk_predictions_run_reference
    ON risk_predictions (scoring_run_id, reference_identifier)
    WHERE scoring_run_id IS NOT NULL;

CREATE INDEX idx_scoring_runs_completed_at
    ON scoring_runs (completed_at DESC);

CREATE INDEX idx_risk_predictions_reference_timestamp
    ON risk_predictions (reference_identifier, prediction_timestamp DESC);

CREATE INDEX idx_risk_predictions_reporting
    ON risk_predictions (prediction_timestamp DESC, risk_band, loan_id);
