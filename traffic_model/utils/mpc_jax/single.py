import functools, math

import numpy as np
import jax, jax.numpy as jnp
import scipy.optimize

from .config import (
    VehicleSpec,
    SolverOption,
    # Dimension
    control_dim,
    # Shape marks
    EgoState,
    ControlHorizon,
    SurStateHorizon,
    ReferenceHorizon,
)
from .ops import ego_rollout

def make_fun(
    vehicle: VehicleSpec, option: SolverOption,
    fun_objective, fun_sur_constraint
):
    dt = option.dt
    horizon = option.horizon
    control_lb, control_ub = option.control_bound
    fun_ego_rollout = functools.partial(ego_rollout, dt=dt, vehicle=vehicle)

    bounds = scipy.optimize.Bounds(np.tile(control_lb, horizon), np.tile(control_ub, horizon))
    def initial_guess(last_control: ControlHorizon):
        x0 = np.zeros_like(last_control)
        x0[:-1,:] = last_control[1:,:]
        x0[-1,:] = last_control[-1,:]
        return np.ravel(x0)
    def fun_min(control_horizon: ControlHorizon, ego_state: EgoState, reference: ReferenceHorizon):
        control_horizon = control_horizon.reshape((horizon, control_dim))
        ego_state_horizon = fun_ego_rollout(ego_state, control_horizon)
        return fun_objective(ego_state_horizon, control_horizon, reference)
    def fun_sur(control_horizon: ControlHorizon, ego_state: EgoState, sur_state_horizon: SurStateHorizon):
        control_horizon = control_horizon.reshape((horizon, control_dim))
        ego_state_horizon = fun_ego_rollout(ego_state, control_horizon)
        return jnp.ravel(fun_sur_constraint(ego_state_horizon, sur_state_horizon))
    # def fun_sur_dot(control_horizon: ControlHorizon, v, ego_state: EgoState, sur_state_horizon: SurStateHorizon):
    #     return jnp.vdot(fun_sur(control_horizon, ego_state, sur_state_horizon), v)

    def make_constraints(ego_state, sur_state_horizon):
        return [scipy.optimize.NonlinearConstraint(
            functools.partial(fun_sur, ego_state=ego_state, sur_state_horizon=sur_state_horizon),
            0.0, math.inf,
            jac=functools.partial(jac_sur, ego_state=ego_state, sur_state_horizon=sur_state_horizon),
            # hess=functools.partial(hess_sur, ego_state=ego_state, sur_state_horizon=sur_state_horizon)
        )]

    def postprocess(x, ego_state):
        control_horizon = np.asarray(x).reshape((horizon, -1))
        ego_state_horizon = np.asarray(fun_ego_rollout(ego_state, control_horizon))
        return control_horizon, ego_state_horizon

    fun_ego_rollout = jax.jit(fun_ego_rollout)
    hess_min = jax.jit(jax.hessian(fun_min))
    fun_min = jax.jit(jax.value_and_grad(fun_min))
    # hess_sur = jax.jit(jax.hessian(fun_sur_dot))
    jac_sur = jax.jit(jax.jacfwd(fun_sur))
    fun_sur = jax.jit(fun_sur)

    return fun_min, hess_min, bounds, initial_guess, make_constraints, postprocess
