SQLITE WORKING STORES AND SEPARATE COMPLETION EVIDENCE — DRAFT, NOT ACTIVATED

Current source SHA-256: 1df73932d15491c1f5f96d007af97f5478fe2f011d4bbbbf5e052258bb493c53 (1,257 files).

SQLite is retained, with no SQL table migration. New completions bind a separate
closed snapshot under runtime/sealed. Engine task databases use SQLite backup;
contracts are checked against the actual snapshot before publication. Working
files may subsequently change without changing sealed evidence. Any change,
addition or removal in the sealed tree remains an evidence mismatch. Historical
completions retain their original exact working-tree checks.

Focused native registered validation: 106 PASS, 0 FAIL, COMPLETED/VERIFIED.
The earlier captured run had 39 PASS and 1 FAIL because the new test module was
missing from qualification/PROFILE.json. That failure remains preserved on Drive.
The successor registers the module and records prospective core-pin successors;
no historical pin or prior result has been rewritten.

This repairs the working/evidence boundary. The process that recreated historical
WAL/SHM sidecars remains unidentified; no root-cause resolution is claimed.
The prior candidate's full suite remains deliberately paused at 527 PASS/0 FAIL.
Those results and its 145-test targeted pass do not qualify this changed source.
Full qualification is deferred to the final release candidate by user instruction.
No activation, release promotion, independent-review claim or scientific promotion.

Use decoder/ as source root. Run python decoder-import/verify_source.py for byte
integrity. REVISION5_QUALIFICATION_RECEIPT.json applies only to revision 5.
GitHub contains source and synthetic tests; run data and checkpoints remain on Drive.
