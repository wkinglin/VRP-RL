

import torch
from rl4co.utils.trainer import RL4COTrainer

from parco.envs import HCVRPEnv
from parco.models import PARCORLModule, PARCOPolicy

import numpy as np
torch.set_printoptions(threshold=np.inf)

# Greedy rollouts over trained model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")# %%
env = HCVRPEnv(generator_params=dict(num_loc=60, num_agents=5),
               data_dir="",
               val_file="../data/hcvrp/n60_m5_seed24610.npz",
               test_file="../data/hcvrp/n60_m5_seed24610.npz",
               )
td_test_data = env.generator(batch_size=[3])
td_init = env.reset(td_test_data.clone()).to(device)
td_init_test = td_init.clone()


emb_dim = 128
policy = PARCOPolicy(env_name=env.name,
                    embed_dim=emb_dim,
                    agent_handler="highprob",
                    normalization="rms",
                    context_embedding_kwargs={
                        "normalization": "rms",
                        "norm_after": False,
                        }, # these kwargs go to the context embed (communication layers)
                    norm_after=False, # True means we use Kool structure
                   )

model = PARCORLModule(  env,
                            policy,
                            train_data_size=10_000,    # Small size for demo
                            val_data_size=1000,       # Small size for demo
                            batch_size=64,              # Small size for demo
                            num_augment=8,              # SymNCO augments to use as baseline
                            train_min_agents=5,         # Minmum number of agents to train on
                            train_max_agents=5,        # Maximum number of agents to train on
                            train_min_size=60,          # Minimum number of locations to train on
                            train_max_size=60,         # Maximum number of locations to train on
                            optimizer_kwargs={'lr': 1e-4, 'weight_decay': 0}, )

trainer = RL4COTrainer(
    max_epochs=20, 
    accelerator="gpu", # change to cpu if you don't have a GPU (note: this will be slow!)
    devices=1, # change this to your GPU number
    logger=None,
)
trainer.fit(model)
trainer.save_checkpoint("./hcvrp_models/hcvrp_rl_model.ckpt")

# Use same initial td as before
model = model.to(device)
out = model(td_init_test.clone(), phase="test", decode_type="greedy", return_actions=True)

# Plotting
actions =  out["actions"]#.reshape(td_init.shape[0], -1)
print("Average tour length: {:.2f}".format(-out['reward'].mean().item()))
for i in range(3):
    print(f"Tour {i} length: {-out['reward'][i].item():.2f}")
    env.render(td_init[i], actions[i].cpu(), plot_number=True)


