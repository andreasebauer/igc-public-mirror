SQLITE SEAL AND ATTEMPT RECOVERY CANDIDATE — NOT ACTIVATED

This draft contains the SQLite completion-sealing repair and lock-bound reconciliation of stale RUNNING attempt records after checkpoint restoration. Original attempt records and digests are retained; mismatched bindings refuse before writes. Historical exact-byte pins remain unchanged, with an explicit prospective successor for the controller change.

Current source SHA-256: a599a49c3882efc5a04f11d23edc100825ec650fedb19f3f36a64f11289cb92f (1,254 files, verified against GitHub).

Validation and attribution:
- Original SQLite-only candidate: full functional run finished 1,060 PASS / 1 FAIL. The stale-attempt failure remains recorded; this is not a full pass.
- Attempt-repair predecessor d4130b53a7820ff0cd3a8d777d6841a0024f113984351cebf11083b58ce439a5: 36 focused PASS and 36 PASS after exact pre-execution checkpoint restoration.
- First combined full capture omitted a required parent fixture and pin successor: 330 PASS / 11 failed checks, paused and preserved. No qualification credit claimed.
- Current corrected source: full rerun deliberately paused after 527 PASS / 0 FAIL, on the user's instruction to avoid repeating expensive algebra checks for this administrative repair.
- Current corrected source: remaining targeted regression completed 145 PASS / 0 FAIL across 17 modules covering locking, recovery, provenance, SQLite lifecycle, completion and validation publication.

The successful registered scientific member replay belongs to the earlier SQLite candidate; no scientific equivalence claim is transferred merely from these counts. Data, failures and recovery objects remain on Drive.

Full-suite qualification of this exact revised source is deferred to the final release candidate. Targeted repair regression is complete. No activation, release promotion, or claim of independent review.

Use decoder/ as source root. Run python decoder-import/verify_source.py for
source-byte verification. REVISION5_QUALIFICATION_RECEIPT.json applies only
to revision 5, not this candidate. Root pyproject belongs to legacy IGC.
Follow native capture/change guides. Preserve independent recovery archives.
Source and synthetic tests live in GitHub; scientific data remain on Drive.
