DECODER SQLITE SEAL AND ATTEMPT RECOVERY CANDIDATE — NOT ACTIVATED
Source identity: d4130b53a7820ff0cd3a8d777d6841a0024f113984351cebf11083b58ce439a5

This branch contains a candidate repair based on dev84 revision 5. The source
manifest verifies its 1,253 files. Run python decoder-import/verify_source.py.
The historical REVISION5_QUALIFICATION_RECEIPT.json applies only to revision 5;
it does NOT qualify this changed candidate. Full qualification remains pending.

Focused native recorded change: 15/15 lifecycle checks passed. Changes switch
completed generation stores out of WAL before sealing, make generation readers
read-only, and strengthen the existing idle partition finalizer to the same
single-file journal mode. Competing readers cause refusal; no journal is deleted
by application code. Original failed evidence is not repaired in place.
SQLite reference: https://www.sqlite.org/wal.html

The actual member-comparison replay completed with VERIFIED evidence. Its data,
witnesses and checkpoints remain in Drive; they are not published here.
The exact cause of the old sidecar reappearance remains unresolved. This is a
verified preventive lifecycle change for the replay, not an exhaustive root-
cause claim or full engineering qualification. No runtime activation.

Use decoder/ as source root; root pyproject belongs to legacy IGC. Follow native
capture/change guides. Preserve independent recovery archives. GitHub contains
source and synthetic regression tests; scientific run data remain on Drive.

Recovery follow-up: restored snapshots can retain a prior RUNNING attempt.
After acquiring exclusive workspace and work locks, the controller retains
that full record and its digest, then marks it INTERRUPTED. Binding mismatches
refuse before any write. Same-PID records across namespaces are not liveness.
Combined native focused qualification: 36 PASS. Exact BEFORE_EXECUTION
checkpoint restoration and replay: 36 PASS. Full functional qualification
is running; expected 1063 tests across 116 selectors. No activation.
