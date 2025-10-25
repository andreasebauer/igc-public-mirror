# GUI Workflow (Decoupled v1)

## Page A — Entry
- Actions:
  - **Create Simulation** → Page B
  - **Run Metrics** → Page C

## Page B — Create Simulation
- Edit/Save a `Simulations` row (D/C/δ/G/K + lattice/time).
- No run here (decoupled). Optional “Resume/Zoom-In” later.

## Page C — Data Selection
- Pick existing run/set of frames (DB-known or index a path).
- Choose frame range; confirm fields present.

## Page D — Metrics Selection
- Table of metrics (steps). User checks rows.
- On selecting a derived metric, parents auto-select and lock.
- Orphans disallowed. No bundles in v1.

## Page F — Run Monitor
- Shows progress (frames × metrics), current job, alerts, events.
- Controls: pause/resume/stop member/stop all.