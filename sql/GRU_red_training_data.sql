WITH GA4_DATA  AS (
  SELECT
    CONCAT('GA', REGEXP_REPLACE(CAST(ga_session_id AS STRING), r'[-.]', '')) AS visit_id,

    event_timestamp  AS vtime,
    REPLACE(
      REGEXP_REPLACE(
          UPPER(REGEXP_REPLACE(page_path_info.page_path_level_2, r'^/|/$', '')), 
          '-', 
          '_'
      ), 
      ' ', ''  -- This removes all spaces
  ) AS brand_raw, 


  REGEXP_REPLACE(
      UPPER(REPLACE(REGEXP_REPLACE(page_path_info.page_path_level_3, r'^/|/$', ''), ' ', '')), 
      '-', 
      '_'
  )  AS product_raw,  

    -- You MUST include event timestamp for order
  TIMESTAMP_MICROS(event_timestamp) AS event_ts
  
  FROM `prod-analytics-derived-c3spc.ga4_enhanced.red`,
    UNNEST(items) AS items,
    UNNEST(event_params) AS ep
  WHERE

    event_date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 12 MONTH)
                    AND CURRENT_DATE()    
    AND page_path_info.page_path IS NOT NULL
    AND REGEXP_CONTAINS(page_path_info.page_path, r'/product')

  ),
  -- Build raw product key per event row (no grouping yet)
step_process_ga AS (
    
    SELECT  DISTINCT
    visit_id, 
    CONCAT(
    FORMAT_TIMESTAMP('%Y-%m-%d', TIMESTAMP_SECONDS(CAST(ROUND(vtime / 1000000 / 60) * 60 AS INT64))),
    "_",
    LPAD(CAST(CASE 
        WHEN EXTRACT(HOUR FROM TIMESTAMP_SECONDS(CAST(ROUND(vtime / 1000000 / 60) * 60 AS INT64))) < 0 THEN 0 
        WHEN EXTRACT(HOUR FROM TIMESTAMP_SECONDS(CAST(ROUND(vtime / 1000000 / 60) * 60 AS INT64))) > 23 THEN 23 
        ELSE EXTRACT(HOUR FROM TIMESTAMP_SECONDS(CAST(ROUND(vtime / 1000000 / 60) * 60 AS INT64))) 
    END AS STRING), 2, '0'),
    "_",
    LPAD(CAST(CASE 
        WHEN EXTRACT(MINUTE FROM TIMESTAMP_SECONDS(CAST(ROUND(vtime / 1000000 / 60) * 60 AS INT64))) < 0 THEN 0 
        WHEN EXTRACT(MINUTE FROM TIMESTAMP_SECONDS(CAST(ROUND(vtime / 1000000 / 60) * 60 AS INT64))) > 59 THEN 59 
        ELSE EXTRACT(MINUTE FROM TIMESTAMP_SECONDS(CAST(ROUND(vtime / 1000000 / 60) * 60 AS INT64))) 
    END AS STRING), 2, '0')
    ) AS vtime, 
    brand_raw,
    product_raw
    FROM(
        SELECT * FROM GA4_DATA         
        
    )
    WHERE brand_raw IS NOT NULL AND product_raw IS NOT NULL AND brand_raw != '' AND product_raw != '' 
      AND REGEXP_CONTAINS(brand_raw, r'[^a-zA-Z]') = false
      AND REGEXP_CONTAINS(product_raw, r'[^a-zA-Z0-9_]') = false
    GROUP BY visit_id,vtime,brand_raw,product_raw
  ),

step_ga AS(
      SELECT visit_id,vtime, CONCAT(brand_raw, "_", product_raw) AS product_key
      FROM step_process_ga
    ),

step_2 AS (
    SELECT
        visit_id,        
        COUNT(DISTINCT product_key) AS visits
    FROM step_ga
    GROUP BY visit_id
    HAVING COUNT(DISTINCT product_key) BETWEEN 2 AND 20
),


step_3 AS (
    SELECT
        a.visit_id AS visit_id,
        product_key,
        vtime,
    FROM step_ga a
    INNER JOIN step_2 b ON a.visit_id = b.visit_id
),
step_4 AS(
  SELECT
  fvisit_id,
  uid
  FROM `prod-analytics-recommend-c1jeg.RED_RECS.GRU_SESSION_LABELS`
  GROUP BY fvisit_id,uid
),
step_5 AS(
  SELECT
  product_id,
  pid
  FROM `prod-analytics-recommend-c1jeg.RED_RECS.GRU_PRODUCT_LABELS`
  WHERE product_id  NOT IN ("XNAT","XNAP","ES","MGT","MWTS","BSAV","DTT","XNAR")
  GROUP BY product_id,pid
)


SELECT
      vtime AS vtime,
      uid AS SessionId,
      pid AS uid,

FROM step_3 AS c
INNER JOIN (
  SELECT *
  FROM step_5) b
  ON c.product_key = b.product_id             
INNER JOIN (
  SELECT *
  FROM step_4) d
  ON c.visit_id = d.fvisit_id 
ORDER BY SessionId, vtime;
