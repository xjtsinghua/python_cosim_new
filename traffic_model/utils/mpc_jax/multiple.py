import functools

import numpy as np
import jax, jax.numpy as jnp
import scipy.optimize

from .config import (
    VehicleSpec,
    SolverOption,
    # Dimension
    state_dim,
    control_dim,
    # Shape marks
    EgoState,
    ControlHorizon,
    SurStateHorizon,
    ReferenceHorizon,
)
from mpc_jax.ops import ego_rollout, vehicle_dynamics

def make_fun(
    vehicle: VehicleSpec, option: SolverOption,
    fun_objective, fun_sur_constraint
):
    dt = option.dt
    horizon = option.horizon
    control_lb, control_ub = option.control_bound
    state_lb, state_ub = option.state_bound
    fun_vehicle_dynamics = jax.vmap(functools.partial(vehicle_dynamics, dt=dt, vehicle=vehicle))

    bounds = scipy.optimize.Bounds(
        np.concatenate((np.tile(control_lb, horizon), np.tile(state_lb, horizon))),
        np.concatenate((np.tile(control_ub, horizon), np.tile(state_ub, horizon)))
    )
    def initial_guess(init_state: EgoState):
        x_dim = (state_dim + control_dim) * horizon
        x0 = np.zeros(x_dim)
        return x0
    def fun_min(state_control_horizon: jnp.ndarray, ego_state: EgoState, reference: ReferenceHorizon):
        control_horizon = state_control_horizon[:horizon * control_dim].reshape((horizon, control_dim))
        ego_state_horizon = state_control_horizon[horizon * control_dim:].reshape((horizon, state_dim))
        return fun_objective(ego_state_horizon, control_horizon, reference)
    def fun_sur(state_control_horizon: jnp.ndarray, ego_state: EgoState, sur_state_horizon: SurStateHorizon):
        # control_horizon = state_control_horizon[:horizon * control_dim].reshape((horizon, control_dim))
        ego_state_horizon = state_control_horizon[horizon * control_dim:].reshape((horizon, state_dim))
        return jnp.ravel(fun_sur_constraint(ego_state_horizon, sur_state_horizon))
    def fun_dynamics(state_control_horizon: jnp.ndarray, ego_state: EgoState):
        control_horizon = state_control_horizon[:horizon * control_dim].reshape((horizon, control_dim))
        ego_state_horizon = state_control_horizon[horizon * control_dim:].reshape((horizon, state_dim))
        ego_state_in = jnp.concatenate((jnp.expand_dims(ego_state, 0), ego_state_horizon[:-1]))
        ego_state_out = ego_state_horizon
        return jnp.ravel(fun_vehicle_dynamics(ego_state_in, control_horizon) - ego_state_out)
    def make_constraints(ego_state, sur_state_horizon):
        return [{
            "type": "ineq",
            "fun": fun_sur,
            "jac": jac_sur,
            "args": (ego_state, sur_state_horizon)
        }, {
            "type": "eq",
            "fun": fun_dynamics,
            "jac": jac_dynamics,
            "args": (ego_state,)
        }]
    def postprocess(x, ego_state):
        x = np.asarray(x).reshape((horizon, control_dim + state_dim))
        control_horizon = x[:, :control_dim]
        ego_state_horizon = x[:, control_dim:]
        return control_horizon, ego_state_horizon

    fun_min = jax.jit(jax.value_and_grad(fun_min))
    jac_sur = jax.jit(jax.jacfwd(fun_sur))
    fun_sur = jax.jit(fun_sur)
    jac_dynamics = jax.jit(jax.jacfwd(fun_dynamics))
    fun_dynamics = jax.jit(fun_dynamics)
    postprocess = jax.jit(postprocess)

    return fun_min, fun_sur, jac_sur, fun_dynamics, jac_dynamics, bounds, initial_guess, make_constraints, postprocess
