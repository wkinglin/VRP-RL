import multiprocessing as mp
import os

# TODO: this is a trick to avoid infinite warnings
# but should be removed in the future
# sys.stderr = open(os.devnull, "w")
# ruff: noqa: E402
import time
import warnings

import torch

# fix for ancdata 0
torch.multiprocessing.set_sharing_strategy("file_system")

from rl4co.data.utils import load_npz_to_tensordict, save_tensordict_to_npz
from tensordict import TensorDict
from tqdm.auto import tqdm

from parco.baselines.ortools.main import solve_and_evaluate
from parco.envs.omdcpdp.env import OMDCPDPEnv

# protobuf warning
warnings.filterwarnings("ignore", category=DeprecationWarning)

mp.set_start_method("spawn", force=True)  # Add this line


# Size to solving time as in paper (seconds)
size_to_time = {
    50: 30,
    100: 60,
    200: 120,
    500: 300,
    1000: 600,
}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_procs", type=int, default=100)
    parser.add_argument("--solver", type=str, default="ortools")
    args = parser.parse_args()
    num_procs = args.num_procs
    solver = args.solver

    # print("Writing output to", f"output_{solver}.txt")
    # sys.stdout = open(f"output_{solver}.txt", "w")
    # use tee instead:
    # python scripts/run_ortools.py --num_procs 64 --solver ortools | tee output_ortools.txt

    print("Solver:", solver, "Num procs:", num_procs)

    env = OMDCPDPEnv(check_solution=False)

    data_files = []
    for root, dirs, files in os.walk("./data/omdcpdp/"):
        for file in files:
            if "sol" not in file and file.endswith(".npz"):  # exclude solution files
                data_files.append(os.path.join(root, file))

    # sort data files by length of name
    data_files = sorted(data_files, key=lambda x: len(x))
    # sort by "val" or "test"
    data_files = sorted(data_files, key=lambda x: x.split("/")[-1])
    # sort by size
    data_files = sorted(
        data_files, key=lambda x: int(x.split("/")[-1].split("n")[1].split("_")[0])
    )

    print(f"Found these files: {data_files}")

    # Go through the files, and run the solver
    for file in tqdm(data_files, desc="Generating dataset solutions with " + solver):
        td_test = load_npz_to_tensordict(file)
        td_test_reset = env.reset(td_test.clone())

        bs = td_test_reset.batch_size[-1]
        m = td_test_reset["capacity"].shape[-1]
        n = td_test_reset["locs"].shape[-2] - m
        is_reset = True

        max_runtime = size_to_time[n]
        print(42 * "=" + "\nProcessing", file, "...")
        print(f"Estimated time : {max_runtime * bs / min(num_procs,bs):.3f} s")

        start = time.time()

        # Main solver
        # Use multiprocessing to solve and evaluate
        with mp.Pool(processes=num_procs) as pool:
            batch_ids = list(range(bs))
            results = pool.starmap(
                solve_and_evaluate,
                [
                    (batch_id, td_test_reset, n, m, max_runtime, is_reset)
                    for batch_id in batch_ids
                ],
            )

        total_time = time.time() - start

        # Post-process for saving
        actions = []
        costs = []
        for r in results:
            actions.append(r["actions"][0])
            costs.append(r["reward"])

        # get max length
        max_len = max([a.shape[-1] for a in actions])

        for i, a in enumerate(actions):
            a = torch.cat(
                [a, a[:, -1].unsqueeze(-1).repeat(1, max_len - a.shape[-1])], dim=-1
            )
            actions[i] = a

        actions = torch.stack(actions)
        costs = torch.tensor(costs)

        print(f"Time: {total_time:.3f} s")
        print(f"Average cost: {costs.mean():.3f}")

        out = TensorDict(
            {
                "actions": actions,
                "costs": costs,
                "time": total_time,
            },
            batch_size=[],
        )

        save_tensordict_to_npz(
            out, file.replace(".npz", f"_sol_{solver}_{max_runtime}s.npz")
        )
