from pathlib import Path
import numpy as np

from igc.sim.grid_constructor import GridConfig, build_humming_grid
from igc.sim.coupler import CouplerConfig, Coupler
from igc.sim.injector import Injector
from igc.sim.integrator import IntegratorConfig, Integrator

# store path; adjust if needed or make env-driven
STORE = Path("/data/igc")
SIM_LABEL = "DEV_pp0"

# 1) Grid
gcfg = GridConfig(shape=(128,128,128))
state = build_humming_grid(gcfg, substeps_per_at=48, initial_at=0)

# 2) Coupler (static, gentle local feedback)
ccfg = CouplerConfig(
    D_psi=0.0, D_eta=0.0, D_phi=0.0,
    C_pi_to_eta=1.0,   # |pi| writes into eta
    C_eta_to_phi=1.0,  # |eta| writes into phi
    lambda_eta=1.0,    # eta decays to ~0 without sustained source
    lambda_phi=1.0,    # phi tends to 0 unless supported by |eta|
    gate="linear"
)
coupler = Coupler(ccfg)

# 3) Injector (pp0 = no injections)
injector = Injector([])

# 4) Integrator
icfg = IntegratorConfig(substeps_per_at=48, dt_per_at=1.0, stride_frames_at=1)
integ = Integrator(coupler, injector, icfg)

# 5) Run 1 At, save start and end
integ.run(store=STORE,
          sim_label=SIM_LABEL,
          psi=state.psi, pi=state.pi, eta=state.eta, phi_field=state.phi_field,
          at_start=0, at_end=1, save_first_frame=True, header_stats=True)

print(f"Done. See {STORE}/Sim_{SIM_LABEL}/Frame_0000 and Frame_0001")
