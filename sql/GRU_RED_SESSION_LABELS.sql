 
--
-- https://cloud.google.com/bigquery/docs/reference/standard-sql/collation-concepts
-- https://support.google.com/analytics/answer/9358801?hl=en
-- 

WITH red_products_ga AS (
    SELECT
    CONCAT('GA', REGEXP_REPLACE(CAST(ga_session_id AS STRING) , r'[-.]', '')) AS visit_id,
    REGEXP_REPLACE(
      UPPER(REGEXP_REPLACE(page_path_info.page_path_level_2, r'^/|/$', '')), 
      '-', 
      '_'
    ) AS brand,
    REGEXP_REPLACE(
      UPPER(REGEXP_REPLACE(page_path_info.page_path_level_3, r'^/|/$', '')), 
      '-', 
      '_'
    ) AS product,  

    FROM `prod-analytics-derived-c3spc.ga4_enhanced.red`,
      UNNEST(items) AS items,
      UNNEST(event_params) AS ep

  WHERE 
    event_date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 12 MONTH)
                    AND CURRENT_DATE()  
      AND page_path_info.page_path IS NOT NULL
      AND REGEXP_CONTAINS(page_path_info.page_path, r'/product')
 
     GROUP BY visit_id,brand,product
  ),
  step_1 AS (
      SELECT 
          visit_id, 
          CONCAT(brand, "_", product) AS product_key,
      FROM red_products_ga

      WHERE brand IS NOT NULL AND product IS NOT NULL 
      AND brand != '' AND product != '' 
      AND REGEXP_CONTAINS(brand, r'[^a-zA-Z]') = false
      AND REGEXP_CONTAINS(product, r'[^a-zA-Z0-9_]') = false
      GROUP BY visit_id, product_key
  ),
  step_2 AS (
  SELECT
    visit_id,
    COUNT(DISTINCT product_key) AS visits
  FROM
    step_1
  GROUP BY
    visit_id
  HAVING
    COUNT(DISTINCT product_key) BETWEEN 2 AND 20)
   
SELECT visit_id AS fvisit_id, ROW_NUMBER() OVER() AS uid FROM step_2
ORDER BY fvisit_id
