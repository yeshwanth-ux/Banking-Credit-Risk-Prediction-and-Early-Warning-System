CREATE TABLE IF NOT EXISTS customers (
    customer_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    customer_reference VARCHAR(64) NOT NULL UNIQUE,
    birth_date DATE,
    annual_income NUMERIC(18, 2),
    employment_status VARCHAR(64),
    country_code VARCHAR(2),
    attributes JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT customers_annual_income_nonnegative
        CHECK (annual_income IS NULL OR annual_income >= 0),
    CONSTRAINT customers_country_code_format
        CHECK (country_code IS NULL OR country_code ~ '^[A-Z]{2}$')
);

CREATE TABLE IF NOT EXISTS loans (
    loan_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    loan_reference VARCHAR(64) NOT NULL UNIQUE,
    customer_id BIGINT NOT NULL,
    application_date DATE NOT NULL,
    origination_date DATE,
    maturity_date DATE,
    principal_amount NUMERIC(18, 2) NOT NULL,
    interest_rate_pct NUMERIC(7, 4),
    term_months INTEGER NOT NULL,
    purpose VARCHAR(128),
    loan_status VARCHAR(32) NOT NULL DEFAULT 'application',
    currency_code VARCHAR(3) NOT NULL DEFAULT 'USD',
    attributes JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT loans_customer_fk
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    CONSTRAINT loans_principal_positive CHECK (principal_amount > 0),
    CONSTRAINT loans_interest_rate_range
        CHECK (interest_rate_pct IS NULL OR interest_rate_pct BETWEEN 0 AND 100),
    CONSTRAINT loans_term_positive CHECK (term_months > 0),
    CONSTRAINT loans_status_allowed
        CHECK (loan_status IN (
            'application', 'approved', 'active', 'delinquent',
            'defaulted', 'paid_off', 'charged_off', 'cancelled'
        )),
    CONSTRAINT loans_currency_format CHECK (currency_code ~ '^[A-Z]{3}$'),
    CONSTRAINT loans_origination_after_application
        CHECK (origination_date IS NULL OR origination_date >= application_date),
    CONSTRAINT loans_maturity_after_origination
        CHECK (
            maturity_date IS NULL
            OR origination_date IS NULL
            OR maturity_date >= origination_date
        )
);

CREATE TABLE IF NOT EXISTS repayment_history (
    repayment_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    loan_id BIGINT NOT NULL,
    installment_number INTEGER NOT NULL,
    due_date DATE NOT NULL,
    payment_date DATE,
    scheduled_amount NUMERIC(18, 2) NOT NULL,
    paid_amount NUMERIC(18, 2) NOT NULL DEFAULT 0,
    days_past_due INTEGER NOT NULL DEFAULT 0,
    repayment_status VARCHAR(24) NOT NULL DEFAULT 'scheduled',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT repayment_history_loan_fk
        FOREIGN KEY (loan_id) REFERENCES loans(loan_id)
        ON UPDATE RESTRICT ON DELETE CASCADE,
    CONSTRAINT repayment_installment_positive CHECK (installment_number > 0),
    CONSTRAINT repayment_scheduled_nonnegative CHECK (scheduled_amount >= 0),
    CONSTRAINT repayment_paid_nonnegative CHECK (paid_amount >= 0),
    CONSTRAINT repayment_days_past_due_nonnegative CHECK (days_past_due >= 0),
    CONSTRAINT repayment_status_allowed
        CHECK (repayment_status IN (
            'scheduled', 'paid', 'partial', 'late', 'missed', 'waived'
        )),
    CONSTRAINT repayment_history_loan_installment_unique
        UNIQUE (loan_id, installment_number)
);

CREATE TABLE IF NOT EXISTS account_activity (
    activity_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    customer_id BIGINT,
    loan_id BIGINT,
    source_event_reference VARCHAR(128),
    activity_timestamp TIMESTAMPTZ NOT NULL,
    activity_type VARCHAR(64) NOT NULL,
    amount NUMERIC(18, 2),
    balance_after NUMERIC(18, 2),
    currency_code VARCHAR(3) NOT NULL DEFAULT 'USD',
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT account_activity_customer_fk
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        ON UPDATE RESTRICT ON DELETE CASCADE,
    CONSTRAINT account_activity_loan_fk
        FOREIGN KEY (loan_id) REFERENCES loans(loan_id)
        ON UPDATE RESTRICT ON DELETE CASCADE,
    CONSTRAINT account_activity_subject_required
        CHECK (customer_id IS NOT NULL OR loan_id IS NOT NULL),
    CONSTRAINT account_activity_currency_format
        CHECK (currency_code ~ '^[A-Z]{3}$'),
    CONSTRAINT account_activity_source_unique UNIQUE (source_event_reference)
);

CREATE TABLE IF NOT EXISTS risk_predictions (
    prediction_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    loan_id BIGINT NOT NULL,
    prediction_timestamp TIMESTAMPTZ NOT NULL,
    model_name VARCHAR(128) NOT NULL,
    model_version VARCHAR(64) NOT NULL,
    default_probability NUMERIC(8, 7) NOT NULL,
    risk_band VARCHAR(16) NOT NULL,
    warning_reasons JSONB NOT NULL DEFAULT '[]'::JSONB,
    feature_snapshot JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT risk_predictions_loan_fk
        FOREIGN KEY (loan_id) REFERENCES loans(loan_id)
        ON UPDATE RESTRICT ON DELETE CASCADE,
    CONSTRAINT risk_predictions_probability_range
        CHECK (default_probability BETWEEN 0 AND 1),
    CONSTRAINT risk_predictions_band_allowed
        CHECK (risk_band IN ('low', 'medium', 'high')),
    CONSTRAINT risk_predictions_warning_reasons_array
        CHECK (jsonb_typeof(warning_reasons) = 'array'),
    CONSTRAINT risk_predictions_feature_snapshot_object
        CHECK (jsonb_typeof(feature_snapshot) = 'object'),
    CONSTRAINT risk_predictions_scoring_unique
        UNIQUE (loan_id, prediction_timestamp, model_name, model_version)
);
