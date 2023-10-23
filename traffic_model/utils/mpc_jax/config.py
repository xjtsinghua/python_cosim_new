from dataclasses import dataclass
import math

import jax.numpy as jnp


EgoState = jnp.ndarray  # Of shape (6,)
EgoStateHorizon = jnp.ndarray  # Of shape (horizon, 6)
Control = jnp.ndarray  # Of shape (2,)
ControlHorizon = jnp.ndarray  # Of shape (horizon, 2)
SurState = jnp.ndarray  # Of shape (4,)
SurStateBatch = jnp.ndarray  # Of shape (N, 4)
SurStateHorizon = jnp.ndarray  # Of shape (N, horizon, 4)
Reference = jnp.ndarray  # Of shape (4,)
ReferenceHorizon = jnp.ndarray  # Of shape (horizon, 4)
Position = jnp.ndarray  # Of shape (2,)
Weight = tuple  # Of shape (7,)
state_dim, control_dim, sur_dim, reference_dim = 6, 2, 4, 4


@dataclass(frozen=True)
class VehicleSpec:
    length: float = 4.5
    width: float = 1.8
    mass: float = 1500.0
    I_zz: float = 2420.0
    a: float = 1.14
    b: float = 1.4
    C_yf: float = -88000.0
    C_yr: float = -94000.0
    u_bound: tuple = (0.0, 100.0)


@dataclass(frozen=True)
class SolverOption:
    dt: float = 0.1
    horizon: int = 10
    sur_count: int = 4
    shooting_mode: str = "single"  # "single" or "multiple"

    # MPC parameters
    weight: tuple = (
        0.0,    # v_x
        4.0,    # x
        4.0,    # y
        100.0,  # phi
        0.0,    # r
        100.0,  # steer
        1.0     # a_x
    )
    control_bound: tuple = (
        (-0.4, -2.0),  # lower bound
        ( 0.4,  1.0)   # upper bound
    )
    state_bound: tuple = (
        (0.0, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf),
        (8.0,  math.inf,  math.inf,  math.inf,  math.inf,  math.inf)
    )   # Only used for multiple shooting
    safety_margin: float = 0.1
    sur_threshold: float = 0.0
    emergency_control: tuple = (0.0, -5.0)  # (steer, a_x)

    # Scipy solver options
    method: str = "SLSQP"
    minimize_options: dict = None

    def __post_init__(self):
        assert self.shooting_mode in ("single", "multiple")
        if self.minimize_options is None:
            # See https://stackoverflow.com/questions/53756788
            object.__setattr__(self, 'minimize_options', SolverOption.default_minimize_options(self.method))

    @staticmethod
    def default_minimize_options(method: str) -> dict:
        if method == "SLSQP":
            return {
                "maxiter": 100,
                "ftol": 1e-4,
                "iprint": 1,
                "disp": False,
            }
        elif method == "trust-constr":
            return {
                "xtol": 1e-4,
                "gtol": 1e-4,
                "barrier_tol": 1e-4,
                "sparse_jacobian": None,
                "maxiter": 100,
                "verbose": 0,
                "finite_diff_rel_step": None,
                "initial_constr_penalty": 1.0,
                "initial_tr_radius": 1.0,
                "initial_barrier_parameter": 0.1,
                "initial_barrier_tolerance": 0.1,
                "factorization_method": None,
                "disp": False
            }
        else:
            return {}
