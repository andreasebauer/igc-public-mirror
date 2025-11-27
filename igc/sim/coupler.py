from dataclasses import dataclass
from typing import Callable, Dict

@dataclass(frozen=True)
class CouplerConfig:
    # Base coefficients (bounded, deterministic)
    D_psi: float = 0.0     # ψ transport
    D_eta: float = 0.0     # η diffusion
    D_phi: float = 0.0     # φ diffusion

    # Couplings and sources
    C_pi_to_eta: float = 1.0   # eta source from |pi|
    C_eta_to_phi: float = 1.0  # phi source from |eta|
    C_psi_to_phi: float = 1.0  # phi source from ψ-gradient skin (|∇ψ|²)
    C_phi_to_psi: float = 1.0  # φ→ψ cone-gated transport strength
    C_psi_to_eta: float = 0.0  # extra eta source from |psi|
    C_eta_to_psi: float = 0.0  # extra psi/π source from eta

    # Decays
    lambda_eta: float = 1.0    # decay of eta
    lambda_phi: float = 1.0    # decay of phi toward 0 (we initialize phi=1 open)
    lambda_psi: float = 0.0    # additional damping on π (default 0: current behaviour)

    # Drive term for π
    gamma_pi: float = 0.0      # constant π drive (default 0: no drive)

    # Restoring spring for ψ (π equation)
    k_psi_restore: float = 1.0

    # Gate function name (for future variants)
    gate: str = "linear"

class Coupler:
    """Provides coefficients per substep (can be static or scheduled)."""

    def __init__(self, cfg: CouplerConfig):
        self.cfg = cfg

    def get(self, at: int, sub: int, tact_phase: float) -> Dict[str, float]:
        # For now, return static constants. Later we can make them time/phase-dependent.
        return {
            "D_psi": self.cfg.D_psi,
            "D_eta": self.cfg.D_eta,
            "D_phi": self.cfg.D_phi,

            "C_pi_to_eta": self.cfg.C_pi_to_eta,
            "C_eta_to_phi": self.cfg.C_eta_to_phi,
            "C_psi_to_phi": self.cfg.C_psi_to_phi,
            "C_phi_to_psi": self.cfg.C_phi_to_psi,
            "C_psi_to_eta": self.cfg.C_psi_to_eta,
            "C_eta_to_psi": self.cfg.C_eta_to_psi,

            "lambda_eta": self.cfg.lambda_eta,
            "lambda_phi": self.cfg.lambda_phi,
            "lambda_psi": self.cfg.lambda_psi,

            "gamma_pi": self.cfg.gamma_pi,
            
            "k_psi_restore": self.cfg.k_psi_restore,
            "gate": self.cfg.gate,
        }
