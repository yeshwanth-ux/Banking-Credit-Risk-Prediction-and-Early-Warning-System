-- Convenience entry point for psql users.
-- The Python migration runner executes files in sql/migrations directly.
\set ON_ERROR_STOP on
BEGIN;
\ir migrations/V001__create_core_schema.sql
\ir migrations/V002__add_operational_indexes.sql
COMMIT;
