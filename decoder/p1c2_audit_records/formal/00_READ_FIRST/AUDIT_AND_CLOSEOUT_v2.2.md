# Formal Foundation v2.2 - Audit and Closeout

**Verdict:** PASS WITH PRECISION CORRECTIONS. No numerical or witness result collapsed. The interrupted pass was scientifically salvageable and is now closed as an audited v2.2 edition.

## Corrections made

1. **Bounded depth and all-finite future were separated.** For a finite depth `d`, `P_d` is the coarsest exact depth-`d` observational quotient; it need not be a reusable transition congruence. The stabilized `P_*` is the all-finite-future reusable quotient. This is the most important conceptual repair because IN11's 2,864-class current-enabledness quotient is intentionally bounded and not transition-congruent.
2. **Set and multiplicity semantics were separated.** Under set-valued successors, `P_*` is ordinary observer-respecting strong bisimulation. Under `Counter` semantics, it is an exact multiplicity-stable transition congruence, not ordinary strong bisimulation.
3. **External bibliography metadata was repaired.** The Park venue was made precise and the Ranzato-Tapparo DOI was added.
4. **The mature observer-resolution census was replayed deterministically.** Its class counts and authoritative lattice result reproduced exactly.

## Results surviving audit

- 7 mature port atoms -> 5 bridge-only atom classes; automorphism group size 4.
- Supplied mature panel: 22,885 exact states, 7,600 port states, 5,241 bridge-multiplicity classes, 64 support/rank-enabled classes, 4 bridge-acceptance classes.
- Bridge multiplicity has the exact five-counter aggregate `(p0+p1,p2,p3,p4,p5+p6)`; a 5x5 response minor is unimodular.
- Template-labelled port children reconstruct parent `p`; template-labelled exact children reconstruct `(p,t)` in the frozen isolated 77-template grammar.
- IN11 current enabledness: A4 3,240 -> exact bounded observer quotient 2,864; A4 sufficient, not minimal.
- O2 Q: actual generated k17 witness proves sitewise `(p,f)` pairing necessary.
- O3 R3: constructive connected witness proves sitewise `(p,f)` pairing inside blocks necessary; rank and internal-relation futures separate the pair.
- Q and R3 remain future-sufficient for their frozen operation algebras; global minimality remains open.

## Replay status

A fresh clean replay reproduced byte-identically the generic theorem-shape tests, mature information lattice and witnesses, mature observer-resolution census, O2 pairing witness and O3 pairing witness. The IN11 computation is substantially heavier; its two archived executions in the salvage checkpoint are byte-identical for both result and witness. A third foreground replay was attempted during this closeout but exceeded the current execution window before completion; no scientific claim depends on that third run.

## Open boundary

The next phase is guarded nonnegative mature reachability, unless global Q/R3 minimality is deliberately prioritized first.

## Certified replay-runner closeout

The final packaging audit found one reproducibility-controller defect: Stage 8 invoked
an alternate aggregate-schema builder rather than the authoritative frozen assembler.
The underlying nine component results/witnesses all reproduced exactly. The controller
was repaired to call `assemble_observer_hierarchy.py`, and resumable fail-closed
checkpointing was added.

A clean staged replay from the included source archives then regenerated all ten frozen
machine-readable outputs under `PYTHONHASHSEED=999`: **10/10 byte-identical, 0
mismatches**. The repair changed no scientific result or claim.
