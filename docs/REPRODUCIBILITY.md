# Reproducibility

Official evidence is tied to a protocol hash, resolved-config hash, architecture hash, initialization hash, dataset/split hashes, environment identity, hardware hash and Git commit. Raw run and telemetry files are immutable. Processed data, tables and figures are regenerated from raw records.

Recommended archival sequence: tag the exact commit, export framework environments, validate all raw records, regenerate outputs, archive split/schedule/initialization artifacts, create a Zenodo release, and add its DOI to `CITATION.cff`.
