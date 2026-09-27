# Reverse Microscope · three-pane focus specification (0.4)

## Instrument surface

| Pane | Visible controls or content | Action |
| --- | --- | --- |
| Left | Current scientific level; Zoom In/Out; Focus; picture magnification | Level changes the source view. Magnification changes the camera only. Focus changes visibility of rejected records. |
| Center | Source-rendered picture of objects and recorded relations | Tap an object or relation to open its source inspector. |
| Right | Observers, match count, individual questions, Ask bundle, Clear answers | Checked questions combine by logical AND within the current level. |

The inspector is a dismissible overlay. Its exact row, source address, and evidence remain available. The underlying bundle retains the All Data library, while the three-pane surface keeps its launcher out of view.

## Question contract

The 0.4 adapter supports the bounded foundation corpus at L0, L1, and L2. Its candidate universe at each level consists of all source objects whose `level` equals that level. Each question is a pure predicate on the saved `row` array `[L, ports, score, destinations, stay]`:

| Question | Recorded predicate |
| --- | --- |
| Label L equals zero | `row[0] === 0` |
| Score s equals two | `row[2] === 2` |
| Stay equals one | `row[4] === 1` |

One checkbox applies one question. **Ask bundle** checks all three at once. **Clear answers** removes all constraints. The denominator always counts unique saved objects at the current level. The numerator counts objects satisfying every checked question. At full Focus, nonmatching objects are hidden in the L-level collection; lower Focus values fade them. Focus never deletes or changes source data. An exact boundary view can return to the collection to apply the same questions.

Questions reset on scientific level changes. No answer to an unchecked question is assumed. The adapter does not treat a match as an ignition, decision, actualization, or physical probability. It does not infer a relation between distinct atlas objects from proximity.

O1–O3 and O4–G8 have source evidence views, but no general per-object observer-answer table in this release. The observer pane labels questions unavailable there; it does not silently filter those records. The future master data block should expose stable object IDs, level, question IDs, answer values (yes/no/unknown), source references, and bundle membership so these same controls can cover L0–G8. Unknown must remain distinct from no; a bundle may narrow candidates or refine distinguishability depending on its declared operation.

## Release verification

- Parse the bundled foundation data and check the three predicates and AND count at L0–L2.
- Check the loader attaches the focus stylesheet and script after the source viewer and upper evidence adapter.
- Exercise tap inspection, level switching, individual and bundled questions, collection return, and narrow-screen layout in a browser.
