CREATE TABLE IF NOT EXISTS loan_prediction_dates (
    loan_id BIGINT NOT NULL,
    prediction_date DATE NOT NULL,
    snapshot_source VARCHAR(32) NOT NULL DEFAULT 'scheduled',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (loan_id, prediction_date),
    CONSTRAINT loan_prediction_dates_loan_fk
        FOREIGN KEY (loan_id) REFERENCES loans(loan_id)
        ON UPDATE RESTRICT ON DELETE CASCADE,
    CONSTRAINT loan_prediction_dates_source_allowed
        CHECK (snapshot_source IN ('application', 'scheduled', 'manual'))
);

CREATE INDEX IF NOT EXISTS idx_loan_prediction_dates_date
    ON loan_prediction_dates (prediction_date);

CREATE OR REPLACE VIEW trusted_loan_analytical_dataset AS
SELECT
    l.loan_id,
    l.loan_reference,
    p.prediction_date,
    p.snapshot_source,
    c.customer_id,
    c.customer_reference,
    c.birth_date,
    CASE
        WHEN c.birth_date IS NULL THEN NULL
        ELSE EXTRACT(YEAR FROM AGE(p.prediction_date, c.birth_date))::INTEGER
    END AS customer_age_years,
    c.annual_income,
    c.employment_status,
    c.country_code,
    l.application_date,
    l.origination_date,
    l.maturity_date,
    l.principal_amount,
    l.interest_rate_pct,
    l.term_months,
    l.purpose,
    l.currency_code,
    COALESCE(r.installments_due, 0)::INTEGER AS installments_due,
    COALESCE(r.installments_paid, 0)::INTEGER AS installments_paid,
    COALESCE(r.installments_overdue, 0)::INTEGER AS installments_overdue,
    COALESCE(r.scheduled_amount_due, 0)::NUMERIC(18, 2) AS scheduled_amount_due,
    COALESCE(r.paid_amount_as_of_date, 0)::NUMERIC(18, 2) AS paid_amount_as_of_date,
    COALESCE(r.outstanding_amount_as_of_date, 0)::NUMERIC(18, 2)
        AS outstanding_amount_as_of_date,
    COALESCE(r.maximum_days_past_due, 0)::INTEGER AS maximum_days_past_due,
    COALESCE(r.has_30_day_delinquency, FALSE) AS has_30_day_delinquency,
    COALESCE(r.has_90_day_delinquency, FALSE) AS has_90_day_delinquency
FROM loan_prediction_dates p
JOIN loans l ON l.loan_id = p.loan_id
JOIN customers c ON c.customer_id = l.customer_id
LEFT JOIN LATERAL (
    SELECT
        COUNT(*) AS installments_due,
        COUNT(*) FILTER (
            WHERE rh.payment_date IS NOT NULL
              AND rh.payment_date <= p.prediction_date
              AND rh.paid_amount >= rh.scheduled_amount
        ) AS installments_paid,
        COUNT(*) FILTER (
            WHERE rh.due_date < p.prediction_date
              AND NOT (
                  rh.payment_date IS NOT NULL
                  AND rh.payment_date <= p.prediction_date
                  AND rh.paid_amount >= rh.scheduled_amount
              )
        ) AS installments_overdue,
        SUM(rh.scheduled_amount) AS scheduled_amount_due,
        SUM(
            CASE
                WHEN rh.payment_date <= p.prediction_date THEN rh.paid_amount
                ELSE 0
            END
        ) AS paid_amount_as_of_date,
        SUM(
            GREATEST(
                rh.scheduled_amount - CASE
                    WHEN rh.payment_date <= p.prediction_date THEN rh.paid_amount
                    ELSE 0
                END,
                0
            )
        ) AS outstanding_amount_as_of_date,
        MAX(
            CASE
                WHEN rh.payment_date IS NOT NULL
                 AND rh.payment_date <= p.prediction_date
                    THEN GREATEST(rh.payment_date - rh.due_date, 0)
                WHEN rh.due_date < p.prediction_date
                    THEN p.prediction_date - rh.due_date
                ELSE 0
            END
        ) AS maximum_days_past_due,
        BOOL_OR(
            CASE
                WHEN rh.payment_date IS NOT NULL
                 AND rh.payment_date <= p.prediction_date
                    THEN rh.payment_date - rh.due_date >= 30
                ELSE p.prediction_date - rh.due_date >= 30
            END
        ) AS has_30_day_delinquency,
        BOOL_OR(
            CASE
                WHEN rh.payment_date IS NOT NULL
                 AND rh.payment_date <= p.prediction_date
                    THEN rh.payment_date - rh.due_date >= 90
                ELSE p.prediction_date - rh.due_date >= 90
            END
        ) AS has_90_day_delinquency
    FROM repayment_history rh
    WHERE rh.loan_id = l.loan_id
      AND rh.due_date <= p.prediction_date
) r ON TRUE
WHERE p.prediction_date >= l.application_date;

CREATE OR REPLACE VIEW trusted_credit_risk_dataset AS
SELECT
    source_record_id,
    checking_account_status,
    duration_months,
    credit_history,
    purpose,
    credit_amount_dm,
    savings,
    employment_duration,
    installment_rate,
    personal_status_sex,
    other_debtors,
    present_residence,
    property,
    age_years,
    other_installment_plans,
    housing,
    number_credits,
    job,
    people_liable,
    telephone,
    foreign_worker,
    bad_credit
FROM credit_risk_observations
WHERE duration_months > 0
  AND credit_amount_dm > 0
  AND age_years BETWEEN 18 AND 120
  AND bad_credit IN (0, 1)
  AND bad_credit = 1 - credit_risk;
