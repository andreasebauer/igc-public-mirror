from pathlib import Path
from igc.sim.grid_constructor import GridConfig, build_humming_grid
from igc.sim.saver import save_frame, HeaderOptions

store = Path("/data/igc")       # adjust if needed
sim_label = "DEV_grid_only"

cfg = GridConfig(shape=(128,128,128))
state = build_humming_grid(cfg, substeps_per_at=48, initial_at=0)

# write frame 0
save_frame(store, sim_label, 0,
           psi=state.psi, pi=state.pi, eta=state.eta, phi_field=state.phi_field,
           at=state.at, substeps_per_at=state.substeps_per_at, tact_phase=state.tact_phase,
           header_opts=HeaderOptions(write_stats=True))
print("Wrote:", (store/f"Sim_{sim_label}/Frame_0000").as_posix())
