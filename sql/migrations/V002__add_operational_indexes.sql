CREATE INDEX IF NOT EXISTS idx_loans_customer_id
    ON loans (customer_id);
CREATE INDEX IF NOT EXISTS idx_loans_application_date
    ON loans (application_date DESC);
CREATE INDEX IF NOT EXISTS idx_loans_status
    ON loans (loan_status);

CREATE INDEX IF NOT EXISTS idx_repayment_history_loan_due_date
    ON repayment_history (loan_id, due_date DESC);
CREATE INDEX IF NOT EXISTS idx_repayment_history_status_due_date
    ON repayment_history (repayment_status, due_date DESC);
CREATE INDEX IF NOT EXISTS idx_repayment_history_payment_date
    ON repayment_history (payment_date DESC)
    WHERE payment_date IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_account_activity_customer_timestamp
    ON account_activity (customer_id, activity_timestamp DESC)
    WHERE customer_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_account_activity_loan_timestamp
    ON account_activity (loan_id, activity_timestamp DESC)
    WHERE loan_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_account_activity_type_timestamp
    ON account_activity (activity_type, activity_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_risk_predictions_loan_timestamp
    ON risk_predictions (loan_id, prediction_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_risk_predictions_timestamp
    ON risk_predictions (prediction_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_risk_predictions_band_timestamp
    ON risk_predictions (risk_band, prediction_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_risk_predictions_model
    ON risk_predictions (model_name, model_version);
