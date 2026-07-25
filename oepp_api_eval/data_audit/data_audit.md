# OEPP API Evaluation Data Audit

Generated: `2026-07-22T10:50:52+00:00`
Data root: `/home/han/research/OEPP/data`

## Paper consistency

| Split | Videos | Segments | Table II match |
|---|---:|---:|---|
| train | 1285 | 5833 | yes |
| validation | 337 | 1479 | yes |
| base | 416 | 1888 | yes |
| novel | 733 | 3010 | yes |

## Sequence windows

| Split | T=3 | T=4 |
|---|---:|---:|
| train | 3268 | 2265 |
| validation | 806 | 546 |
| base | 1058 | 722 |
| novel | 1544 | 958 |

## Action pools

| Pool | Entries | Unique | Duplicate entries |
|---|---:|---:|---:|
| base | 122 | 122 | 0 |
| novel | 55 | 55 | 0 |
| total | 161 | 161 | 0 |

Base/Novel string overlap: **16**.

## Integrity findings

- **train:** 0 ground-truth actions outside its protocol pool; 0 declared-length mismatches; 0 duplicate `(dataset, vid)` identities.
- **validation:** 0 ground-truth actions outside its protocol pool; 0 declared-length mismatches; 0 duplicate `(dataset, vid)` identities.
- **base:** 0 ground-truth actions outside its protocol pool; 0 declared-length mismatches; 0 duplicate `(dataset, vid)` identities.
- **novel:** 0 ground-truth actions outside its protocol pool; 0 declared-length mismatches; 0 duplicate `(dataset, vid)` identities.

## Visual asset readiness

- Image files under data root: **0**.
- Image files under repository root: **1**.
- Annotation fields containing image/frame paths: **0**.
- Visual manifest ready: **false**.

The current repository cannot produce a valid visual API manifest: annotations identify timestamps and action segments, but do not identify raw frame files. Supply a source-backed observation index before running `build_manifest.py`.
