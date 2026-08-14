-- ============================================================================
-- churn_analysis.sql — churn drivers & model-driven risk analytics
-- Database: data/processed/telco_churn.db  (build with: python -m src.build_database)
-- Run interactively:  sqlite3 -column -header data/processed/telco_churn.db
--
-- predictions.is_test_set = 1 marks the held-out 20%: model scores there are
-- honest out-of-sample numbers. Performance-style queries filter to it;
-- operational ranking queries (contact lists) use all customers.
-- ============================================================================

-- Q8. Churn rate by contract type — the strongest single association.
SELECT
    Contract,
    COUNT(*)                                                          AS customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END)                    AS churned,
    ROUND(AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0 END) * 100, 1)  AS churn_rate_pct
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;

-- Q9. Churn rate by internet service.
SELECT
    InternetService,
    COUNT(*)                                                          AS customers,
    ROUND(AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0 END) * 100, 1)  AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2)                                     AS avg_monthly_charges
FROM customers
GROUP BY InternetService
ORDER BY churn_rate_pct DESC;

-- Q10. Churn by number of protective add-ons (internet customers only) —
-- the "support services associate with retention" finding, in SQL.
SELECT
    num_protective                                                    AS protective_addons,
    COUNT(*)                                                          AS customers,
    ROUND(AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0 END) * 100, 1)  AS churn_rate_pct
FROM customers
WHERE InternetService != 'No'
GROUP BY num_protective
ORDER BY num_protective;

-- Q11. HIGH-VALUE / HIGH-RISK: flagged customers with above-median bills —
-- the priority outreach list. Joins model scores to customer attributes.
SELECT
    c.customerID,
    p.churn_probability,
    p.risk_segment,
    c.MonthlyCharges,
    c.tenure,
    c.Contract,
    c.InternetService
FROM customers c
JOIN predictions p USING (customerID)
WHERE p.risk_segment = 'High'
  AND c.MonthlyCharges > (SELECT AVG(MonthlyCharges) FROM customers)
ORDER BY c.MonthlyCharges * p.churn_probability DESC   -- expected monthly revenue at risk
LIMIT 15;

-- Q12. Top 20 customers ranked by churn risk (operational contact list).
-- ROW_NUMBER() gives an explicit rank. Still-subscribed customers only.
SELECT
    ROW_NUMBER() OVER (ORDER BY p.churn_probability DESC)  AS risk_rank,
    c.customerID,
    ROUND(p.churn_probability, 3)                          AS churn_probability,
    c.tenure, c.Contract, c.InternetService, c.PaymentMethod, c.MonthlyCharges
FROM customers c
JOIN predictions p USING (customerID)
WHERE c.Churn = 'No'
ORDER BY p.churn_probability DESC
LIMIT 20;

-- Q13. Risk-segment summary on the HELD-OUT TEST SET (honest performance view):
-- size, actual churn, model calibration, revenue profile per tier.
SELECT
    p.risk_segment,
    COUNT(*)                                                             AS customers,
    ROUND(AVG(p.churn_probability) * 100, 1)                             AS avg_predicted_pct,
    ROUND(AVG(CASE WHEN c.Churn = 'Yes' THEN 1.0 ELSE 0 END) * 100, 1)   AS actual_churn_pct,
    ROUND(AVG(c.MonthlyCharges), 2)                                      AS avg_monthly_charges,
    ROUND(AVG(c.tenure), 1)                                              AS avg_tenure_months
FROM customers c
JOIN predictions p USING (customerID)
WHERE p.is_test_set = 1
GROUP BY p.risk_segment
ORDER BY avg_predicted_pct;

-- Q14. Annualized revenue at risk by segment (test set): what is financially
-- exposed in each tier — sizing input for a retention budget.
SELECT
    p.risk_segment,
    COUNT(*)                                                    AS customers,
    SUM(CASE WHEN c.Churn = 'Yes' THEN 1 ELSE 0 END)            AS actual_churners,
    ROUND(SUM(CASE WHEN c.Churn = 'Yes'
                   THEN c.MonthlyCharges * 12 ELSE 0 END), 0)   AS annualized_churned_revenue
FROM customers c
JOIN predictions p USING (customerID)
WHERE p.is_test_set = 1
GROUP BY p.risk_segment
ORDER BY annualized_churned_revenue DESC;
