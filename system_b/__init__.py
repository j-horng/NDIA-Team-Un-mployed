"""
System B — Imagery Server/Cache

- Scans `data/tiles/{z}/{x}/{y}.json` (metadata) and `{...}.png` (imagery)
- Serves /imagery?lat&lon&zoom (PNG bytes) with X-Geo-Metadata header (JSON)
- Optional endpoints: /dem, /nearest, /stats, /health
"""
