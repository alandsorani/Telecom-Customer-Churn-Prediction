-- ============================================================================
-- customer_analysis.sql — customer base & service analytics
-- Database: data/processed/telco_churn.db  (build with: python -m src.build_database)
-- Run interactively:  sqlite3 -column -header data/processed/telco_churn.db
-- Tables: customers (cleaned + engineered), predictions (model scores)
-- ============================================================================

-- Q1. Overall churn rate.
-- The AVG(CASE ...) idiom turns a Yes/No label into a rate in one pass.
SELECT
    COUNT(*)                                              AS customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END)        AS churned,
    ROUND(AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0 END) * 100, 2) AS churn_rate_pct
FROM customers;

-- Q2. Customer counts and churn by tenure group.
-- tenure_group is the engineered lifecycle bucket. Ordering by MIN(tenure)
-- keeps the buckets in lifecycle order without a lookup table.
SELECT
    tenure_group,
    COUNT(*)                                              AS customers,
    ROUND(AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0 END) * 100, 1) AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2)                         AS avg_monthly_charges
FROM customers
GROUP BY tenure_group
ORDER BY MIN(tenure);

-- Q3. Average charges by churn status — the revenue profile of leavers vs stayers.
SELECT
    Churn,
    COUNT(*)                        AS customers,
    ROUND(AVG(MonthlyCharges), 2)   AS avg_monthly,
    ROUND(AVG(TotalCharges), 2)     AS avg_total,
    ROUND(AVG(tenure), 1)           AS avg_tenure_months
FROM customers
GROUP BY Churn;

-- Q4. Service adoption among internet customers, with churn rate per service.
-- UNION ALL builds a compact adoption/retention scoreboard.
SELECT 'OnlineSecurity' AS service,
       ROUND(AVG(CASE WHEN OnlineSecurity = 'Yes' THEN 1.0 ELSE 0 END) * 100, 1) AS adoption_pct,
       ROUND(AVG(CASE WHEN OnlineSecurity = 'Yes' AND Churn = 'Yes' THEN 1.0
                      WHEN OnlineSecurity = 'Yes' THEN 0 END) * 100, 1)          AS churn_pct_with_service
FROM customers WHERE InternetService != 'No'
UNION ALL
SELECT 'TechSupport',
       ROUND(AVG(CASE WHEN TechSupport = 'Yes' THEN 1.0 ELSE 0 END) * 100, 1),
       ROUND(AVG(CASE WHEN TechSupport = 'Yes' AND Churn = 'Yes' THEN 1.0
                      WHEN TechSupport = 'Yes' THEN 0 END) * 100, 1)
FROM customers WHERE InternetService != 'No'
UNION ALL
SELECT 'OnlineBackup',
       ROUND(AVG(CASE WHEN OnlineBackup = 'Yes' THEN 1.0 ELSE 0 END) * 100, 1),
       ROUND(AVG(CASE WHEN OnlineBackup = 'Yes' AND Churn = 'Yes' THEN 1.0
                      WHEN OnlineBackup = 'Yes' THEN 0 END) * 100, 1)
FROM customers WHERE InternetService != 'No'
UNION ALL
SELECT 'DeviceProtection',
       ROUND(AVG(CASE WHEN DeviceProtection = 'Yes' THEN 1.0 ELSE 0 END) * 100, 1),
       ROUND(AVG(CASE WHEN DeviceProtection = 'Yes' AND Churn = 'Yes' THEN 1.0
                      WHEN DeviceProtection = 'Yes' THEN 0 END) * 100, 1)
FROM customers WHERE InternetService != 'No'
UNION ALL
SELECT 'StreamingTV',
       ROUND(AVG(CASE WHEN StreamingTV = 'Yes' THEN 1.0 ELSE 0 END) * 100, 1),
       ROUND(AVG(CASE WHEN StreamingTV = 'Yes' AND Churn = 'Yes' THEN 1.0
                      WHEN StreamingTV = 'Yes' THEN 0 END) * 100, 1)
FROM customers WHERE InternetService != 'No'
UNION ALL
SELECT 'StreamingMovies',
       ROUND(AVG(CASE WHEN StreamingMovies = 'Yes' THEN 1.0 ELSE 0 END) * 100, 1),
       ROUND(AVG(CASE WHEN StreamingMovies = 'Yes' AND Churn = 'Yes' THEN 1.0
                      WHEN StreamingMovies = 'Yes' THEN 0 END) * 100, 1)
FROM customers WHERE InternetService != 'No'
ORDER BY adoption_pct DESC;

-- Q5. Payment-method mix: share of base, churn rate, and revenue exposure.
SELECT
    PaymentMethod,
    COUNT(*)                                                          AS customers,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customers), 1)     AS pct_of_base,
    ROUND(AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0 END) * 100, 1)  AS churn_rate_pct,
    ROUND(SUM(MonthlyCharges), 0)                                     AS monthly_revenue
FROM customers
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;

-- Q6. Segmentation matrix: contract x internet service.
-- The SQL twin of the EDA heatmap (notebook 03, figure 10).
SELECT
    Contract,
    InternetService,
    COUNT(*)                                                          AS customers,
    ROUND(AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0 END) * 100, 1)  AS churn_rate_pct
FROM customers
GROUP BY Contract, InternetService
ORDER BY churn_rate_pct DESC;

-- Q7. High-value customers: top revenue quartile, with their churn exposure.
-- NTILE(4) window function assigns revenue quartiles without hardcoding cutoffs.
WITH ranked AS (
    SELECT
        customerID, MonthlyCharges, tenure, Contract, Churn,
        NTILE(4) OVER (ORDER BY MonthlyCharges) AS revenue_quartile
    FROM customers
)
SELECT
    revenue_quartile,
    COUNT(*)                                                          AS customers,
    ROUND(MIN(MonthlyCharges), 2)                                     AS min_monthly,
    ROUND(MAX(MonthlyCharges), 2)                                     AS max_monthly,
    ROUND(AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0 END) * 100, 1)  AS churn_rate_pct
FROM ranked
GROUP BY revenue_quartile
ORDER BY revenue_quartile;
