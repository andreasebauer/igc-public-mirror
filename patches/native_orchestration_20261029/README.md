# Native orchestration repair — 29 October 2026

This review branch carries a patch against the recovered deployed source archive whose SHA-256 is recorded in `SOURCE_BINDINGS.json`. The repository's older `decoder/` tree is unchanged. Apply this patch only to that exact baseline; `files/` contains the complete changed and added text files.

The repair keeps execution in the native chain: `rolling_operator.resume_registered_capture` → shared `bulk_submission._run_saved_capture` → registered controller → native stage runtime. It adds no external scheduler or launch wrapper.

Changes include explicit save-wait liveness and interruption recovery, streaming raw-readback checks through the native batch confirmer, typed native-source and accepted-result save boundary continuation, failure exit status for unsuccessful completions, observable bounded source/runtime hashing before validation, honest progress freshness, and recovery of diagnostic histories without arbitrary historical file-count refusal. The capture guide documents native project import/adopt after restoring capture bytes.

Regression tests accompany the changes. This source mirror does not claim scientific acceptance, completed J3 calculations, process survival after environment loss, or promotion of the old repository source tree. The parent audit records final qualification and deployment evidence separately.

To reconstruct, verify the baseline archive SHA-256, extract it into an empty directory, apply `source.patch` with `patch -p1`, and verify every resulting changed-file SHA-256 from `SOURCE_BINDINGS.json`. No source, runtime, receipt, or scientific result pin should be rewritten in an existing frozen capture.
