DECODER GITHUB IMPORT — 26 September 2026

decoder/ contains the byte-exact 0.8.0.dev84+lib revision 2 candidate.
This is a source import, not release promotion or full qualification.
The predecessor dev83 revision 6 will also be preserved in Git history.

Verified source identity:
1484b9f5637099c60773db8ecf66de887daa52686e68bebb81844c765e74764c
Verified package identity:
33ce2fe699a673741870959cceff27f27e3b0fd3b157fef362a673beceb6931b
Original archive SHA-256:
19920b0ce80a23460d42b095bfb524277aefc51e738251bd5a2b4c4837cfa8fe
Original archive: https://drive.google.com/file/d/14CeJxcLlraYqSAg6nSDITSkYi6K0vFc7/view

Predecessor source identity:
b0630fefadd08281c0babb9f0bcb9852e5e9c9e61326b87abfc0ff45369801ad
Predecessor archive SHA-256:
009922cc44ce8bebaa71d060901d7bd6de4d9a0d5e1caaf8ca83830bf020591d
Original archive: https://drive.google.com/file/d/18vWL718846Rk_9J-WFyLJknJSSYzCIY6/view

Recorded qualification (saved evidence, not rerun here):
DEV84 revision 2: 16 focused checks PASS; original A30 benchmark PASS once.
Full DEV84 qualification pending. DEV83 timing failed its original threshold.
DEV83 remains the recorded local development pointer. Neither source import
changes the Decoder's shared or task-local runtime pointers.
Evidence: https://drive.google.com/file/d/1fvodE-Csrsfg9-REnAoPInwXbxAhzPVH/view
Evidence archive SHA-256:
c6b4b4fa8bab72d5ff6ea2238dbbd27ff371ee75dbccfaf70640004df428b758
All 428 members of that bundle were checked against its manifest.

Only three source paths differ between these two snapshots:
infinity_grid/_version.py
infinity_grid/exact_tree_relation_kernel.py
tests/test_compact_label_reuse.py

HOW TO USE
Clone this repository and select an explicit commit/branch.
Run: python decoder-import/verify_source.py
Work from decoder/ as the source root, not the repository root. The root
pyproject.toml belongs to the older IGC orchestrator.
Use Linux CPython 3.12 and decoder/qualification/PROFILE.json for the saved
qualification environment. The current local environment is not claimed to
match that profile. Installation alone is not qualification.
Read decoder/DECODER_06_CAPTURE_GUIDE.txt, DECODER_06_CHANGES_GUIDE.txt and
DECODER_V07_CONTRIBUTOR_WORKFLOW.txt before changing or running the Decoder.
Scientific work and qualification go through its registered native controller.
Do not substitute an unregistered pytest invocation or transfer old receipts
to changed source. Preserve failures and interruption history.

This import includes every original source-archive member, including fixtures,
resources and historical packaging metadata. Even the historical egg-info is
retained because removing it changes the recorded source identity. Read version
from infinity_grid/_version.py. New import/storage documentation lives outside
decoder/ so it does not alter either source identity.
Large runtime objects and full recovery closures remain external in Drive.
A checkout is a complete source snapshot, not a complete science workspace.

DATA STORAGE
See shared-data/README.txt. A prepared S3-compatible store design keeps hashes
and scientific identities independent of the storage provider. No remote data
migration or deletion is claimed by this import.
