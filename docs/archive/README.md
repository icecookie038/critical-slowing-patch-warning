# Historical recovery status — 2026-09-05

## Verified file in this repository

`stage_summary_2026-09-04.docx` is the unchanged user-provided stage summary
retrieved from the cloud project's uploaded attachment. SHA256:
`fc7fdcfc75d21feb01698c64b05fe939253781b0e1fb07ebac19b213488238ba`.

The document records the v3.5–v3.7 transition from spatial early-warning claims
toward real-data recovery explanation and time-safe event prediction. It reports
negative/mixed incremental patch results; these should not be replaced by the
new prototype's engineering or pilot results.

## Source archive still pending transfer

The original cloud task, **重新分析调整方案**, reports creating commit
`9473acb724ff67c9780f7a80c4cdc01c135c157f` on
`archive/v3.7-recovered-2026-09-05`, based on
`945804cf54c34d54bb017ec0e132ee2acbd7f462`. It reports 60 changed files and
16 historical unit tests passing in that source environment. Neither that
commit nor those tests have yet been independently verified in this checkout.
The cloud branch has not been uploaded to GitHub.

The source reports these checksums:

- Original complete export ZIP:
  `5b6c3506ea499c8b4918ab66fb5eeffbc604d18746aa31c461b1c7259baba155`.
- Incremental Git bundle, 1,308,511 bytes:
  `31f98cf9d86ffb846cbb19f812fa75029b0cd586f4ba19bfac242c69028ee53a`.
- Source-only tar.gz, 76,835 bytes:
  `1425c09410b874ee469c0751f03711020b196258c7f00abf6c24801fc8db4f9e`.

The source-only package contains 29 source/test/script/document/JSON files and
excludes 25 CSV files and six PNG figures. It is not a complete archive.
These are source-reported hashes, not proof of local receipt.

Local authenticated Git push succeeds. The cloud GitHub integration returns
HTTP 403 `Resource not accessible by integration`; its generated attachment is
not exposed as a directly downloadable local file by the task-reading interface.
An explicitly authorized text transfer has been requested but not yet received
and verified. No credentials have been copied between tasks.

Completion requires receiving the complete ZIP or Git bundle, verifying its
hash and paths, importing the original files, running the historical tests,
and pushing a separately identified archive commit. Until then this repository
contains the latest stage summary and the new QIR prototype, **not a complete
v3.7 source restoration**.
