import multiprocessing

import numpy as np
import torch


# from mdopdptw.problem import Problem
from rl4co.data.utils import load_npz_to_tensordict

from parco.baselines.ortools.solver import OrtoolsSolver, Problem
from parco.envs import OMDCPDPEnv

# Configuration
solver_type = "ortools"
dir_name = "ortools_600s"
SOLVER = {"ortools": OrtoolsSolver}
SOLVER_PARAM = {
    "ortools": {
        "vehicle_max_travel": 500000,  # max travel distance
        "use_soft_time_windows": True,
        "lateness_penalty": 200,
    }
}
timelimit = 300  # 481
ms = {1000: [100, 250, 500]}

# Instantiate the solver
solver = SOLVER[solver_type]()


# Convert route to action
def convert_route_to_action(route, num_tasks, num_agents):
    actions = []
    max_route = max([len(r) for r in route])
    for sub_route in route:
        action = []
        for i, r in enumerate(sub_route):
            if i == 0:
                action.append(r - num_tasks)
            else:
                action.append(r + num_agents)
        last_action = action[-1]
        for _ in range(max_route - len(sub_route)):
            action.append(last_action)
        actions.append(action)
    return torch.tensor(actions, dtype=torch.int64).unsqueeze(0)


# Evaluate the solution
def evaluate(td_, actions, env):
    td = td_.clone()
    for i in range(actions.shape[-1]):
        td.set("action", actions[..., i])
        td = env.step(td)["next"]
    rewards = env.get_reward(td.clone(), actions)
    return -rewards.mean().item()


# Define the function to run in parallel
def solve_and_evaluate(
    batch_id, td_loaded, n=100, m=10, timelimit=timelimit, is_reset=False
):
    env = OMDCPDPEnv(generator_params=dict(num_loc=n, num_agents=m))
    if not is_reset:
        reset_td = env.reset(td_loaded.clone())
        locs_tensor = td_loaded["locs"][batch_id]
    else:
        num_agents = td_loaded["depots"][batch_id].shape[-2]
        locs_tensor = td_loaded["locs"][batch_id][num_agents:]
        reset_td = td_loaded.clone()

    capacity_tensor = td_loaded["capacity"][batch_id]
    depots_tensor = td_loaded["depots"][batch_id]

    N = locs_tensor.shape[-2]
    M = depots_tensor.shape[-2]
    locs_all = torch.cat([locs_tensor, depots_tensor], 0) * 100
    travel_time_mtx_tensor = torch.abs(
        locs_all.unsqueeze(0) - locs_all.unsqueeze(1)
    ).norm(p=2, dim=-1)
    travel_time_mtx_tensor[:, -m:] = 0  # open trip

    travel_time_mtx = travel_time_mtx_tensor.tolist()
    earliest_time = [0] * (N + M)
    latest_time = [9999] * (N + M)  # large number
    service_time = [0] * (N + M)
    setup_time = [0] * (N + M)
    demand = [1] * (N // 2) + [-1] * (N // 2) + [0] * M
    # capacity = [3] * M
    capacity = capacity_tensor.tolist()  # TODO: check if correct
    max_tasks = [(N // 2) // M * 2 + 2] * M
    n_pickup = N // 2
    n_dropoff = N // 2
    coords = tuple(locs_all.T.tolist())
    fixed = [-1] * (N + M)
    secured = [[]] * M

    # Setup problem instance
    problem = Problem(
        travel_time_mtx=travel_time_mtx,
        earliest_time=earliest_time,
        latest_time=latest_time,
        service_time=service_time,
        setup_time=setup_time,
        demand=demand,
        capacity=capacity,
        max_tasks=max_tasks,
        n_pickup=n_pickup,
        n_dropoff=n_dropoff,
        coords=coords,
        fixed=fixed,
        secured=secured,
        n=N + M,
        n_vehicles=M,
    )

    # Solve the problem
    routes, obj = solver.solve(
        problem=problem, **SOLVER_PARAM[solver_type], timelimit=timelimit
    )

    # Convert routes to actions
    actions = convert_route_to_action(routes, num_tasks=N, num_agents=M)

    # Evaluate the solution
    reward = evaluate(reset_td.clone()[batch_id].unsqueeze(0), actions, env)

    return {"reward": reward, "actions": actions}


# Main process
# TODO: check
if __name__ == "__main__":
    for n in ms.keys():
        for m in ms[n]:
            filename = f"./data/omdcpdp/seoul_n{n}_m{m}.npz"
            td_loaded = load_npz_to_tensordict(filename)
            bs = td_loaded.batch_size[-1]

            # Use multiprocessing to solve and evaluate
            pool = multiprocessing.Pool(processes=min(multiprocessing.cpu_count(), bs))
            batch_ids = list(range(bs))
            results = pool.starmap(
                solve_and_evaluate,
                [(batch_id, td_loaded, n, m) for batch_id in batch_ids],
            )

            # Print the results
            print(f"timelimit: {timelimit}, ag: {m}")
            # print(f"Average reward: {np.mean(results)}")
            print(f"Average reward: {np.mean([r['reward'] for r in results])}")
