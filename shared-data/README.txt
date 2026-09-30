SUPERSEDED STORAGE PROPOSAL — NOT ACTIVE
Current decision: keep data on Google Drive. R2 was not adopted.
For Decoder use ../DECODER_READ_FIRST.txt and ../decoder-admin/CATALOG.json.
The following proposal is retained as history, not current instructions.

SHARED INFINITY GRID DATA STORE — PROTOTYPE, NOT YET PROVISIONED
26 September 2026

Decision proposed: one authoritative S3-compatible object store, with a private
scientific archive and a published read-only collection for the Microscope.
Use Cloudflare R2 Standard as the first candidate. AWS S3 is an alternative.
Source code remains in GitHub. Keep Drive as a recovery backup during migration.
One authoritative store does not mean deleting the independent backup.

WHY THIS FITS THE INSPECTED SOURCE SNAPSHOTS
Decoder preservation already names many objects by SHA-256. Keep those exact
bytes and hashes. A downloaded object is only a transport input: the Decoder's
native verification, role checks, source binding and restore still apply.
Do not mount object storage as a running workspace: qualification/PROFILE.json
requires local POSIX locks, hard links, fsync and atomic rename. Run locally,
publish closed immutable objects, restore locally on the next machine/chat.

The inspected Microscope baseline (commit 4c2a04d) contains a 9,322,589-byte JSON pack embedded in a gzip
HTML application split into 20 base64 files. Its own Load another bundle action
already accepts the same JSON schema. The staging utility extracts the exact
JSON bytes, without changing records, IDs, provenance, or claimed coverage.
shared-data.js is a tested standalone transport prototype for pinned HTTPS
objects. It is deliberately NOT wired into the Microscope: another chat updated
main to a newer multi-pack viewer during this import. Its current interface must
be reconciled before integration. No Microscope file is changed by the final PR.
The staging pilot extracts the legacy embedded pack, not the new complete viewer
catalogue. Do not mistake the pilot for a current or exhaustive corpus export.
No cloud account, charge or public release is created.

LAYOUT AND IDENTITY
objects/sha256/<first-two-hex>/<full-sha256> = exact immutable file bytes.
A catalogue row names the hash, byte length, role, schema and provenance.
Content identity is separate from provider URL; relocation does not rename
scientific objects. Identical bytes are uploaded once. Changed bytes get new keys.
Catalogue snapshots are immutable and versioned. A small current pointer moves
only after all referenced objects are available and verified.
Microscope packs are derived views, not replacements for canonical carriers.
Preserve their source-object hashes, transformation version and scope limits.
There is currently NO certified L2-to-SCOUT3 crosswalk; storage cannot create it.

ACCESS
Keep source captures, runtime history and unreviewed evidence private.
Publish only explicitly selected scientific views already reviewed for release.
Use a public read-only bucket/custom domain for those views, or an authenticated
gateway if the complete corpus must share one physical private bucket. A public
R2 bucket exposes its contents; CORS is not an authorization mechanism.
Use bucket-scoped write credentials only in trusted server/CLI environments.
Never put write credentials or signed private download links in browser code,
GitHub manifests or source. Preserve a protected backup and retention policy.

PUBLICATION AND RECOVERY GATES
1. Inventory existing Drive objects and classify canonical / checkpoint / view /
   source / historical failure. Do not discard old evidence by filename/date.
2. Deduplicate exact bytes by SHA-256. Preserve dependency roles and manifests.
3. Upload missing immutable objects with a standard S3 client (rclone/AWS CLI).
   Never use destructive sync or delete-after-transfer during this migration.
4. Raw-download every newly published object and verify its size and SHA-256.
   Do not substitute ETag, object name or HEAD metadata for byte verification.
5. Verify every catalogue dependency before publishing the catalogue pointer.
6. Restore an existing saved Decoder completion with the native controller and
   prove reuse without redispatch. Load the same published view in the Microscope
   and check exact bytes, schema, IDs and provenance.
7. Only after those gates change the authoritative storage pointer. Retain Drive.

PREPARED LOCAL EXPORT
Run from the repository root:
  python shared-data/prepare_store.py --output /absolute/path/outside-checkout
This verifies the imported Decoder, inventories its resources and fixtures,
extracts the public Microscope JSON, and stages deduplicated objects plus a
catalogue. It performs no science, native preservation acknowledgement, network
upload or source-resource removal. A staging manifest is not a certified dataset.
The Microscope descriptor is also usable by decoder-side ingestion tooling via
ordinary HTTPS GET followed by SHA-256 validation. Decoder scientific admission
must remain native; this task does not add a new execution path.

BEFORE CLOUD CUTOVER
An owner-controlled R2 account/bucket and authorized upload connection are needed.
No Cloudflare plugin was found in this session. Choose the account, region policy,
private archive/public-view access model and a public custom domain. Configure
GET/HEAD CORS for https://andreasebauer.github.io and expose ETag if desired.
For production public delivery, use a custom domain; r2.dev is rate-limited
development access. Then set microscope/data-source.json base_url to that HTTPS
endpoint and its microscope descriptor to the staged catalogue's view record
only AFTER adapting and testing the latest viewer. The example configuration is
shared-data/data-source.example.json; it is not an active viewer setting.
The loader checks length and SHA-256 before passing JSON to the existing viewer.
Incorrect or unavailable remote data fails visibly; it does not silently show a
different dataset. The existing offline service worker needs a separate reviewed
remote-object cache policy before promising offline use of the remote view.

COST SNAPSHOT (not an invoice; checked 26 September 2026)
R2 Standard: $0.015/GB-month; first 10 GB-month free. About 50 GB stored for a
month implies $0.60 storage after that allowance, plus billable operations and
any gateway/domain/tax costs. Egress bandwidth is not charged by R2. Actual
deduplicated corpus size has NOT been measured; 50 GB is an earlier estimate.
Storage is strongly consistent and designed for high durability; this does not
protect against all deletion/account mistakes, so retain a separate backup.

Provider references:
https://developers.cloudflare.com/r2/pricing/
https://developers.cloudflare.com/r2/how-r2-works/
https://developers.cloudflare.com/r2/buckets/public-buckets/
https://developers.cloudflare.com/r2/buckets/cors/
https://developers.cloudflare.com/r2/buckets/bucket-locks/

CURRENT LIMITS
No full Drive corpus inventory, cloud migration, live remote-store integration,
native cold-restore proof from R2, or whole-branch scientific qualification has
been completed by this import. Existing required package resources remain byte
exact in decoder/ until separately qualified external-resource loading exists.
