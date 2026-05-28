SELECT
  
  CONCAT(key_brand, "_", UPPER(key_product)) AS product_id,
  ROW_NUMBER() OVER() AS pid
FROM
  `expanded-nebula-754.staging.sial_catalog_*`
WHERE
  _TABLE_SUFFIX = (SELECT MAX(_TABLE_SUFFIX) FROM `expanded-nebula-754.staging.sial_catalog_*`)
  AND key_brand IS NOT NULL AND key_product IS NOT NULL AND key_brand != '' AND key_product != '' 
  AND REGEXP_CONTAINS(key_brand, r'[^a-zA-Z]') = false
  AND REGEXP_CONTAINS(key_product, r'[^a-zA-Z0-9_]') = false
  AND UPPER(key_product_status) NOT IN ("WITHDRAWN","INTERNAL")
  

GROUP BY
  product_id