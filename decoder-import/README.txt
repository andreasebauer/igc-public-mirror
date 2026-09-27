DECODER SQLITE SEAL AND ATTEMPT RECOVERY CANDIDATE — NOT ACTIVATED
Source identity: a599a49c3882efc5a04f11d23edc100825ec650fedb19f3f36a64f11289cb92f

This branch contains a candidate repair based on dev84 revision 5. The source
manifest verifies its 1,254 files. Run python decoder-import/verify_source.py.
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

Qualification correction: the first full capture omitted its required parent
snapshot fixture and the controller byte-pin successor. It was paused and
its failed checks retained. A corrected capture includes the exact historical
parent fixture and an explicit prospective pin successor. No production code
changed in this correction; no historical pin or assertion was removed.
The 36+36 focused results above refer to the preceding candidate identity.
Full qualification of this corrected identity is now running; not qualified.
