import math, functools, itertools

import jax, jax.numpy as jnp

from .config import (
    VehicleSpec,
    # Shape marks
    EgoState,
    EgoStateHorizon,
    Control,
    ControlHorizon,
    SurState,
    Reference,
    ReferenceHorizon,
    Position,
    Weight,
)


def vehicle_dynamics(
    state: EgoState, control: Control,
    dt: float, vehicle: VehicleSpec
) -> EgoState:
    v_x, v_y, r, x, y, phi = state
    steer, a_x = control
    mass, I_zz, a, b, C_yf, C_yr, (u_lb, u_ub) = vehicle.mass, vehicle.I_zz, vehicle.a, vehicle.b, vehicle.C_yf, vehicle.C_yr, vehicle.u_bound
    next_state = jnp.array((
        jnp.clip(v_x + dt * (a_x + v_y * r), u_lb, u_ub),
        (mass * v_y * v_x + dt * (a * C_yf - b * C_yr) * r - dt * C_yf * steer * v_x - dt * mass * v_x ** 2 * r) / (mass * v_x - dt * (C_yf + C_yr)),
        (-I_zz * r * v_x - dt * (a * C_yf - b * C_yr) * v_y + dt * a * C_yf * steer * v_x) / (dt * (a ** 2 * C_yf + b ** 2 * C_yr) - I_zz * v_x),
        x + dt * (v_x * jnp.cos(phi) - v_y * jnp.sin(phi)),
        y + dt * (v_x * jnp.sin(phi) + v_y * jnp.cos(phi)),
        phi + dt * r
    ))
    return next_state


def angle_wrap(angle: float) -> float:
    """rad -> [-pi, pi]"""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def sur_dynamics(
    state: SurState,
    dt: float
) -> SurState:
    x, y, v, phi = state
    x_delta = v * dt * jnp.cos(phi)
    y_delta = v * dt * jnp.sin(phi)
    phi_rad_delta = 0
    next_state = jnp.array((
        x + x_delta,
        y + y_delta,
        v,
        phi + phi_rad_delta
    ))
    return next_state


def objective_one_step(
    state: EgoState, control: Control, reference: Reference,
    weight: Weight
) -> float:
    v_x, v_y, r, x, y, phi = state
    steer, a_x = control
    x_r, y_r, phi_r, v_xr = reference
    return (
        weight[0] * (v_x - v_xr)            ** 2 +
        weight[1] * (x - x_r)               ** 2 +
        weight[2] * (y - y_r)               ** 2 +
        weight[3] * angle_wrap(phi - phi_r) ** 2 +
        weight[4] * r                       ** 2 +
        weight[5] * steer                   ** 2 +
        weight[6] * a_x                     ** 2
    )


def objective_horizon(
    ego_state: EgoStateHorizon, control: ControlHorizon,
    reference: ReferenceHorizon, weight: Weight
) -> float:
    fun_objective_one_step = functools.partial(objective_one_step, weight=weight)
    fun_objective = jax.vmap(fun_objective_one_step)
    return jnp.sum(fun_objective(ego_state, control, reference))


def position_threshold_constraint(
    ego: Position, other: Position,
    threshold: float
) -> float:
    # scipy requires inequality constraint to be non-negative
    distance = jnp.sqrt(jnp.sum(jnp.square(ego - other)))
    return distance - threshold


def get_constraint_positions(
    center: Position, phi: float,
    offset: float
) -> Position:
    dcm = jnp.array((jnp.cos(phi), jnp.sin(phi)))
    return (
        center + dcm * offset,
        center - dcm * offset
    )


def sur_constraint(
    ego: EgoState, other: SurState,
    margin: float, vehicle: VehicleSpec
) -> jnp.ndarray:
    length, width = vehicle.length, vehicle.width
    ego_offset = other_offset = (length - width) / 2
    ego_radius = other_radius = math.hypot(width / 2, width / 2)
    threshold = ego_radius + other_radius + margin
    ego_positions = get_constraint_positions(ego[3:5], ego[5], ego_offset)
    other_positions = get_constraint_positions(other[:2], other[3], other_offset)
    return jnp.min(jnp.stack([
        position_threshold_constraint(ego_position, other_position, threshold)
        for ego_position, other_position in itertools.product(ego_positions, other_positions)
    ]))


def ego_rollout(
    init_state: EgoState, control: ControlHorizon,
    dt: float, vehicle: VehicleSpec
) -> EgoStateHorizon:
    def dynamics(state: SurState, control: Control) -> SurState:
        next_state = vehicle_dynamics(state, control, dt, vehicle)
        return next_state, next_state
    _, state_horizon = jax.lax.scan(dynamics, init_state, control)
    return state_horizon


def sur_rollout(
    init_state: SurState,
    dt: float, horizon: int
) -> SurState:
    def dynamics(state: SurState, _: None) -> SurState:
        next_state = sur_dynamics(state, dt)
        return next_state, next_state
    _, state_horizon = jax.lax.scan(dynamics, init_state, xs=None, length=horizon)
    return state_horizon
