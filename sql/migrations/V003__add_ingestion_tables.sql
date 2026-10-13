CREATE TABLE IF NOT EXISTS ingestion_runs (
    ingestion_run_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_name VARCHAR(255) NOT NULL,
    source_sha256 CHAR(64) NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ,
    status VARCHAR(16) NOT NULL DEFAULT 'running',
    rows_read INTEGER NOT NULL DEFAULT 0,
    rows_accepted INTEGER NOT NULL DEFAULT 0,
    rows_rejected INTEGER NOT NULL DEFAULT 0,
    rows_inserted INTEGER NOT NULL DEFAULT 0,
    rows_updated INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    CONSTRAINT ingestion_runs_status_allowed
        CHECK (status IN ('running', 'completed', 'failed')),
    CONSTRAINT ingestion_runs_counts_nonnegative CHECK (
        rows_read >= 0 AND rows_accepted >= 0 AND rows_rejected >= 0
        AND rows_inserted >= 0 AND rows_updated >= 0
    )
);

CREATE TABLE IF NOT EXISTS credit_risk_observations (
    observation_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_record_id CHAR(64) NOT NULL UNIQUE,
    source_row_number INTEGER NOT NULL,
    checking_account_status SMALLINT NOT NULL,
    duration_months INTEGER NOT NULL,
    credit_history SMALLINT NOT NULL,
    purpose SMALLINT NOT NULL,
    credit_amount_dm NUMERIC(18, 2) NOT NULL,
    savings SMALLINT NOT NULL,
    employment_duration SMALLINT NOT NULL,
    installment_rate SMALLINT NOT NULL,
    personal_status_sex SMALLINT NOT NULL,
    other_debtors SMALLINT NOT NULL,
    present_residence SMALLINT NOT NULL,
    property SMALLINT NOT NULL,
    age_years INTEGER NOT NULL,
    other_installment_plans SMALLINT NOT NULL,
    housing SMALLINT NOT NULL,
    number_credits SMALLINT NOT NULL,
    job SMALLINT NOT NULL,
    people_liable SMALLINT NOT NULL,
    telephone SMALLINT NOT NULL,
    foreign_worker SMALLINT NOT NULL,
    credit_risk SMALLINT NOT NULL,
    bad_credit SMALLINT NOT NULL,
    ingestion_run_id BIGINT NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT observations_ingestion_run_fk FOREIGN KEY (ingestion_run_id)
        REFERENCES ingestion_runs(ingestion_run_id) ON DELETE RESTRICT,
    CONSTRAINT observations_duration_positive CHECK (duration_months > 0),
    CONSTRAINT observations_amount_positive CHECK (credit_amount_dm > 0),
    CONSTRAINT observations_age_reasonable CHECK (age_years BETWEEN 18 AND 120),
    CONSTRAINT observations_target_binary CHECK (credit_risk IN (0, 1)),
    CONSTRAINT observations_bad_credit_binary CHECK (bad_credit IN (0, 1)),
    CONSTRAINT observations_target_inverse CHECK (bad_credit = 1 - credit_risk)
);

CREATE TABLE IF NOT EXISTS ingestion_rejections (
    rejection_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ingestion_run_id BIGINT NOT NULL,
    source_row_number INTEGER,
    source_record JSONB NOT NULL,
    error_codes JSONB NOT NULL,
    rejected_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT ingestion_rejections_run_fk FOREIGN KEY (ingestion_run_id)
        REFERENCES ingestion_runs(ingestion_run_id) ON DELETE CASCADE,
    CONSTRAINT ingestion_rejections_record_object
        CHECK (jsonb_typeof(source_record) = 'object'),
    CONSTRAINT ingestion_rejections_codes_array
        CHECK (jsonb_typeof(error_codes) = 'array')
);

CREATE INDEX IF NOT EXISTS idx_observations_ingestion_run
    ON credit_risk_observations (ingestion_run_id);
CREATE INDEX IF NOT EXISTS idx_ingestion_runs_started_at
    ON ingestion_runs (started_at DESC);
CREATE INDEX IF NOT EXISTS idx_ingestion_rejections_run
    ON ingestion_rejections (ingestion_run_id);
