-- PostgreSQL 14 end-to-end test for Deletion-Compliant SHI Learned Index (pg_shi_index)

CREATE OR REPLACE FUNCTION shi_build_from_query(
    index_name text,
    sql_query text,
    epsilon int4 DEFAULT 16,
    anchor_modulus int4 DEFAULT 4096,
    mode text DEFAULT 'optimal'
) RETURNS text
AS :'shi_lib_path', 'shi_build_from_query'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION shi_lookup(
    index_name text,
    search_key int8
) RETURNS int8
AS :'shi_lib_path', 'shi_lookup'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION shi_delete_key(
    index_name text,
    target_key int8
) RETURNS text
AS :'shi_lib_path', 'shi_delete_key'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION shi_audit_erasure(
    index_name text,
    n_probes int4 DEFAULT 200
) RETURNS text
AS :'shi_lib_path', 'shi_audit_erasure'
LANGUAGE C STRICT;

DROP TABLE IF EXISTS gdpr_users;
CREATE TABLE gdpr_users (
    user_id int8 PRIMARY KEY,
    account_tier int4 NOT NULL
);

-- Populate 200,000 strictly increasing non-uniform 64-bit user IDs
INSERT INTO gdpr_users (user_id, account_tier)
SELECT
    (i * 1000 + ((i * 2654435761) % 997) + 100000)::int8 AS user_id,
    (i % 5)::int4 AS account_tier
FROM generate_series(1, 200000) AS g(i);

\echo '--- 1. Build Canonical SHI-PLA Index inside PostgreSQL 14 ---'
SELECT shi_build_from_query(
    'users_shi_idx',
    'SELECT user_id FROM gdpr_users ORDER BY user_id',
    16,
    4096,
    'optimal'
)::jsonb AS build_report;

\echo '--- 2. Verify Exact Learned Index Lookups in SQL ---'
SELECT
    u.user_id,
    shi_lookup('users_shi_idx', u.user_id) AS learned_rank
FROM (
    SELECT user_id FROM gdpr_users ORDER BY user_id LIMIT 5
) u;

\echo '--- 3. Execute GDPR Article 17 Right-to-Erasure on Single Key ---'
DO $$
DECLARE
    v_key int8;
    v_res text;
BEGIN
    SELECT user_id INTO v_key FROM gdpr_users ORDER BY user_id OFFSET 50000 LIMIT 1;
    DELETE FROM gdpr_users WHERE user_id = v_key;
    v_res := shi_delete_key('users_shi_idx', v_key);
    RAISE NOTICE 'Single-key GDPR erasure result: %', v_res;
END $$;

\echo '--- 4. Run 200-Delete GDPR Erasure & T1 Leakage Audit inside PostgreSQL 14 ---'
SELECT shi_audit_erasure('users_shi_idx', 200)::jsonb AS gdpr_audit_report;
