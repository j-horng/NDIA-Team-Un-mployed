## GeoFix (C â†’ D)
ts (ISO8601), lat, lon, alt_m?, R (3x3 m^2), confidence [0..1], inliers, rmse_px

## /imagery (B)
GET /imagery?lat&lon&zoom â†’ image/png with X-Geo-Metadata (JSON: geotransform, size, CRS)
