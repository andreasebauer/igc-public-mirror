from dataclasses import dataclass
from typing import Callable, Dict

@dataclass(frozen=True)
class CouplerConfig:
    # Base coefficients (bounded, deterministic)
    D_psi: float = 0.0     # transport disabled in this minimal integrator pass
    D_eta: float = 0.0     # (we'll keep simple local updates first)
    D_phi: float = 0.0
    C_pi_to_eta: float = 1.0  # eta source from |pi|
    C_eta_to_phi: float = 1.0 # phi source from |eta|
    lambda_eta: float = 1.0   # decay of eta
    lambda_phi: float = 1.0   # decay of phi toward 0 (we initialize phi=1 open)

    # Gate function name (for future variants)
    gate: str = "linear"

class Coupler:
    """Provides coefficients per substep (can be static or scheduled)."""

    def __init__(self, cfg: CouplerConfig):
        self.cfg = cfg

    def get(self, at: int, sub: int, tact_phase: float) -> Dict[str, float]:
        # For now, return static constants. In the next passes, you can add schedules here.
        return {
            "D_psi": self.cfg.D_psi,
            "D_eta": self.cfg.D_eta,
            "D_phi": self.cfg.D_phi,
            "C_pi_to_eta": self.cfg.C_pi_to_eta,
            "C_eta_to_phi": self.cfg.C_eta_to_phi,
            "lambda_eta": self.cfg.lambda_eta,
            "lambda_phi": self.cfg.lambda_phi,
            "gate": self.cfg.gate,
        }
