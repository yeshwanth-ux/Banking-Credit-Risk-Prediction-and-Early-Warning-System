ALTER TABLE scoring_runs
    ADD COLUMN idempotency_key VARCHAR(255),
    ADD CONSTRAINT scoring_runs_idempotency_key_nonempty
        CHECK (idempotency_key IS NULL OR length(trim(idempotency_key)) > 0),
    ADD CONSTRAINT scoring_runs_idempotency_key_unique
        UNIQUE (idempotency_key);
