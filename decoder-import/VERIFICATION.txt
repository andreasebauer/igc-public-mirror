TWO-SORTED RELATION LIBRARY ADAPTER — DRAFT, NOT ACTIVATED

Current source SHA-256: c03fdf0e43d0440ef4ac25ab6e43a64da73be44049070e1797c818ccf5821d8d (1,264 files).

Adds a separate whole-relation igraph/BLISS adapter with shared port and destination slots, fixed row/value colors, and exact graph plus row witness replay. Small tuple/set composition preserves all lawful rows and destination concatenation. Existing owner-graph adapter remains separate.

Focused registered Decoder validation: 22 PASS, 0 FAIL, COMPLETED/VERIFIED.
Includes 14 new adapter tests, four existing owner-adapter tests and four qualification-contract tests. Archived 45 relational vectors use the new isomorphism/composition adapters; 203 composition calls agree exactly with the pinned reference and 124 library isomorphism calls run. Three archived cospan families also pass. Source pins identify unchanged historical code. This is bounded implementation evidence, not proof of generic recursive completeness or primitive exactness.

Operational refusal limits: 8 slots per sort, 4096 rows, 50000 gadget vertices, 1000000 composition pairs. These are resource guards, not exhaustive qualification of every input inside them. No new canonical row naming or production routing.

Completion: d54fab51b04253a1460b0a003ab22bef0a5a3acf591c896bcf9a1830a2896fc3.
The inherited SQLite separate-evidence change previously passed 106 focused tests on source 1df73932d15491c1f5f96d007af97f5478fe2f011d4bbbbf5e052258bb493c53. This turn did not repeat that suite. Its historical sidecar cause remains unidentified. A later tuple/list completion-publication mismatch remains a separate open obligation.

Full-suite qualification remains deferred to the release candidate. No activation, merge, canonical science promotion or independent-review claim.

Use decoder/ as source root; decoder-import/verify_source.py checks byte integrity. The revision-5 qualification receipt applies only to revision 5. Source and synthetic test vectors belong here; scientific datasets and execution checkpoints stay on Drive.
