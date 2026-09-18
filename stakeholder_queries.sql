-- Self-service examples for non-technical baseball questions.

-- What is the track record of each country segment?
SELECT segment AS country,
       n,
       ROUND(100.0 * outcome_rate, 1) AS positive_outcome_pct,
       ROUND(100.0 * ci_low, 1) AS ci_low_pct,
       ROUND(100.0 * ci_high, 1) AS ci_high_pct
FROM v_archetype_track_record
WHERE dimension = 'country_std'
ORDER BY outcome_rate DESC, n DESC;

-- Which player records require data-steward review?
SELECT source, rule, severity, COUNT(*) AS flagged_rows
FROM v_data_quality_queue
GROUP BY source, rule, severity
ORDER BY CASE severity WHEN 'error' THEN 1 ELSE 2 END, flagged_rows DESC;

-- Illustrative projection board with scouting context.
SELECT player_id,
       player_name,
       country_std,
       league,
       position,
       age,
       bonus_tier,
       performance_index,
       projected_next_index,
       projection_percentile,
       overall_future_value
FROM v_player_projection_board
ORDER BY projection_percentile DESC;

