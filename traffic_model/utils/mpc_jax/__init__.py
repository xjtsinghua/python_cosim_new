def _setup():
    import jax
    jax.config.update('jax_platform_name', 'cpu')
_setup()

from .solver import make_solver as make_solver
from .config import (
    VehicleSpec as VehicleSpec,
    SolverOption as SolverOption,
    # Dimension
    state_dim as state_dim,
    control_dim as control_dim,
    sur_dim as sur_dim,
    reference_dim as reference_dim,
)
