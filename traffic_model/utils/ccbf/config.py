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


# @dataclass(frozen=True)
# class VehicleSpec:
#     length: float = 4.5
#     width: float = 1.8
#     mass: float = 1500.0
#     I_zz: float = 2420.0
#     a: float = 1.14
#     b: float = 1.4
#     C_yf: float = -88000.0
#     C_yr: float = -94000.0
#     u_bound: tuple = (0.0, 100.0)

# Vehicle spec for CCBF
@dataclass(frozen=True)
class VehicleSpec:
    length: float = 4.5
    width: float = 1.8
    mass: float = 1412.0
    I_zz: float = 1536.7
    a: float = 1.06
    b: float = 1.85
    C_yf: float = -128915.5
    C_yr: float = -85943.6
    u_bound: tuple = (0.0, 100.0)

@dataclass(frozen=True)
class CCBFOption:
    dt: float = 0.1
    horizon: int = 10
    sur_count: int = 4
    
    # CCBF parameters
    # controller.update_lyapunov_parameter(P1, P2, P3, P4, P5, P6)
    weight: tuple = (
        0.0,    # x
        0.1,    # y =0.1 kim推荐0.3,0.5
        0.1,  # phi 0.5 0.1
        0.5,    # vx  0.5
        0.1,  # vy  0.2
        0.5     # omega
    )
    weight1:tuple=(0, 1.5, 2.4, 0.8, 0.00382303, 0.00148199)    #(0, 0.8, 3.0, 0.8, 0.01, 0.001)这两组参数是0阶优化得到的，可以二选一
    
    control_bound: tuple = (
        (-0.2, -3.0),  # lower bound -0.4,-2
        ( 0.2,  2)   # upper bound 0.4 1
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
