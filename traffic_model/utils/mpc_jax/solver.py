import functools, warnings

import numpy as np
import jax, jax.numpy as jnp
import scipy.optimize

from .config import (
    VehicleSpec,
    SolverOption,
    # Dimension
    state_dim,
    control_dim,
    reference_dim,
    # Shape marks
    EgoState,
    ControlHorizon,
    SurStateBatch,
    SurStateHorizon,
    Reference,
    ReferenceHorizon,
)
from .ops import (
    vehicle_dynamics,
    objective_horizon,
    sur_constraint,
    ego_rollout,
    sur_rollout,
)

def make_solver(vehicle: VehicleSpec, option: SolverOption):
    # Options
    dt = option.dt
    horizon = option.horizon
    minimize_method = option.method
    minimize_options = option.minimize_options
    shooting_mode = option.shooting_mode
    weight = option.weight
    safety_margin = option.safety_margin
    sur_threshold = option.sur_threshold
    emergency_control = option.emergency_control

    # Rollout
    fun_ego_rollout = functools.partial(ego_rollout, dt=dt, vehicle=vehicle)
    fun_sur_rollout = jax.vmap(functools.partial(sur_rollout, dt=dt, horizon=horizon))
    fun_sur_constraint = jax.vmap(
        jax.vmap(functools.partial(sur_constraint,margin=safety_margin, vehicle=vehicle)),  # horizon dimension
        in_axes=(None, 0)                                                                   # batch dimension
    )
    fun_objective = functools.partial(objective_horizon, weight=weight)

    # Shooting mode
    if shooting_mode == "single":
        from .single import make_fun as make_fun_single
        fun_min, hess_min, bounds, initial_guess, make_constraints, postprocess = make_fun_single(vehicle, option, fun_objective, fun_sur_constraint)
    elif shooting_mode == "multiple":
        from .multiple import make_fun as make_fun_multiple
        fun_min, bounds, initial_guess, make_constraints = make_fun_multiple(vehicle, option, fun_objective, fun_sur_constraint)
    else:
        raise ValueError("Unknown shooting mode: {}".format(shooting_mode))

    # Jax jit
    fun_ego_rollout = jax.jit(fun_ego_rollout)
    fun_sur_rollout = jax.jit(fun_sur_rollout)
    fun_sur_constraint = jax.jit(fun_sur_constraint)
    
    # Solve function
    def solve(ego_state: EgoState, sur_state: SurStateBatch, reference: Reference, last_control: ControlHorizon) -> ControlHorizon:
        sur_state_horizon = fun_sur_rollout(sur_state)
        constraints = make_constraints(ego_state, sur_state_horizon)
        sln: scipy.optimize.OptimizeResult = scipy.optimize.minimize(
            fun=fun_min,
            x0=initial_guess(last_control),
            args=(ego_state, reference),
            jac=True,
            # hess=hess_min,
            bounds=bounds,
            constraints=constraints,
            method=minimize_method,
            options=minimize_options
        )
        control_horizon, ego_state_horizon = postprocess(sln["x"], ego_state)
        # Safety shield
        sur_condition = fun_sur_constraint(ego_state_horizon, sur_state_horizon)
        if np.all(sur_condition >= sur_threshold):
            if not sln["success"]:
                warnings.warn(f"Solver failed but safety shield is satisfied: {sln['message']}")
            return control_horizon, ego_state_horizon
        else:
            warnings.warn(f"Solver failed:\n{sln}\nEgo: {ego_state}\nSur: {sur_state}\nRef: {reference}")
            control_horizon = np.zeros((horizon, control_dim))
            control_horizon[0] = emergency_control
            ego_state_horizon = np.asarray(fun_ego_rollout(ego_state, control_horizon))
            return control_horizon, ego_state_horizon

    return solve


if __name__ == '__main__':
    vehicle_spec = VehicleSpec()
    solver_option = SolverOption(shooting_mode="single")
    solve = make_solver(vehicle_spec, solver_option)

    ego_state = np.zeros((state_dim,))
    sur_state = np.tile(np.array((100.0, 0.0, 0.0, 0.0)), (10, 1))
    reference = np.zeros((40, reference_dim))
    reference[:, 0] = np.arange(40)
    reference[:, 3] = 1.0
    last_control = np.zeros((solver_option.horizon, control_dim))
    control_horizon, ego_state_horizon = solve(ego_state, sur_state, reference, last_control)
    print(control_horizon, ego_state_horizon)
