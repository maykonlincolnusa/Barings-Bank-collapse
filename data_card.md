# Data Card

## Data sources

- Curated public-source registry in `sources.csv`
- Normalized public event corpus in `app/data/raw/public/barings_public_extracts.json`
- Proxy market data in `app/data/raw/public/market_nikkei_jgb_1995.csv`
- Synthetic scenario outputs in `app/data/synthetic`

## Sensitive data

None. The prototype uses only public references and synthetic records.

## Synthetic design

The simulator reproduces patterns documented in public Barings materials:

- front and back office overlap
- use of hidden account `88888`
- funding transfers to support margin
- delayed reconciliation and backdated entries
- loss escalation after the January 1995 Kobe earthquake shock

## Known gaps

- Real SIMEX/Osaka trade-level data is not bundled.
- Public-source ingestion is curated and normalized rather than full-text mirrored.

