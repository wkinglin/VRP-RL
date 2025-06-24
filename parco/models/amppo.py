from random import randint
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rl4co.data.transforms import StateAugmentation
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.zoo.amppo import AMPPO
from rl4co.models.zoo.symnco.losses import invariance_loss, solution_symmetricity_loss
from rl4co.utils.ops import unbatchify
from rl4co.utils.pylogger import get_pylogger
from torchrl.modules.models import MLP
from rl4co.models.rl.common.critic import CriticNetwork

from .utils import resample_batch

log = get_pylogger(__name__)


class PARCOAMPPOModule(AMPPO):
    """RL LightningModule for PARCO based on AMPPO with multi-agent support"""

    def __init__(
        self,
        # Base AMPPO parameters
        env: RL4COEnvBase,
        policy: nn.Module = None,
        critic: CriticNetwork = None,
        policy_kwargs: dict = {},
        critic_kwargs: dict = {},
        # PARCO parameters
        use_projection_head: bool = True,
        projection_head: nn.Module = None,
        num_augment: int = 4,
        alpha: float = 0.2,
        beta: float = 1.0,
        train_min_agents: int = 5,
        train_max_agents: int = 15,
        train_min_size: int = 50,
        train_max_size: int = 100,
        val_test_num_agents: int = 10,
        allow_multi_dataloaders: bool = True,
        **kwargs,
    ):
        # Initialize the parent AMPPO
        super().__init__(
            env=env,
            policy=policy,
            critic=critic,
            policy_kwargs=policy_kwargs,
            critic_kwargs=critic_kwargs,
            **kwargs
        )

        print(f"Amppo init")
        log.info("Amppo init")


        # PARCO specific parameters
        self.use_projection_head = use_projection_head
        self.num_augment = num_augment
        self.augment = StateAugmentation(num_augment=self.num_augment)
        self.alpha = alpha  # weight for invariance loss
        self.beta = beta    # weight for solution symmetricity loss

        # Create projection head if needed
        if self.use_projection_head:
            if projection_head is None:
                embed_dim = self.policy.decoder.embed_dim
                projection_head = MLP(embed_dim, embed_dim, 1, embed_dim, nn.ReLU)
            self.projection_head = projection_head

        # Multiagent training parameters
        self.train_min_agents = train_min_agents
        self.train_max_agents = train_max_agents
        self.train_min_size = train_min_size
        self.train_max_size = train_max_size
        self.val_test_num_agents = val_test_num_agents
        self.allow_multi_dataloaders = allow_multi_dataloaders

        # Force env to have as num_agents and num_locs as the maximum number of agents and locations
        self.env.generator.num_loc = self.train_max_size
        self.env.generator.num_agents = self.train_max_agents

    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        # NOTE: deprecated
        num_agents = None  # done inside the sampling
        # Sample number of agents during training step
        if phase == "train":
            # Idea: we always have batches of the same size from the dataloader.
            # however, here we sample a subset of agents and locations from the batch.
            # For instance: if we have always 10 depots and 100 cities, we sample a random number of depots and cities
            # from the batch. This way, we can train on different number of agents and locations.
            num_agents = randint(self.train_min_agents, self.train_max_agents)
            num_locs = randint(self.train_min_size, self.train_max_size)
            batch = resample_batch(batch, num_agents, num_locs)
        else:
            if self.allow_multi_dataloaders:
                # Get number of agents to test on based on dataloader name
                if dataloader_idx is not None and self.dataloader_names is not None:
                    # e.g. n50_m7 take only number after "m" until _
                    num_agents = int(
                        self.dataloader_names[dataloader_idx].split("_")[-1][1:]
                    )
                else:
                    num_agents = self.val_test_num_agents

                # NOTE: trick: we subsample number of agents by setting num_agents
                # in such case, use the same number of agents for all batches
                batch["num_agents"] = torch.full(
                    (batch.shape[0],), num_agents, device=batch.device
                )

        # Reset env based on the number of agents
        with torch.no_grad():
            td = self.env.reset(batch)
            out = self.policy(td.clone(), self.env, phase=phase)

        if phase == "train":
            batch_size = out["actions"].shape[0]

            # infer batch size
            if isinstance(self.ppo_cfg["mini_batch_size"], float):
                mini_batch_size = int(batch_size * self.ppo_cfg["mini_batch_size"])
            elif isinstance(self.ppo_cfg["mini_batch_size"], int):
                mini_batch_size = self.ppo_cfg["mini_batch_size"]
            else:
                raise ValueError("mini_batch_size must be an integer or a float.")

            if mini_batch_size > batch_size:
                mini_batch_size = batch_size

            td.set("logprobs", out["log_likelihood"])
            td.set("reward", out["reward"])
            td.set("actions", out["actions"])

            # Inherit the dataset class from the environment for efficiency
            dataset = self.env.dataset_cls(td)
            dataloader = DataLoader(
                dataset,
                batch_size=mini_batch_size,
                shuffle=True,
                collate_fn=dataset.collate_fn,
            )

            for _ in range(self.ppo_cfg["ppo_epochs"]):  # PPO inner epoch, K
                for sub_td in dataloader:
                    sub_td = sub_td.to(td.device)
                    previous_reward = sub_td["reward"].view(-1, 1)
                    out = self.policy(  # note: remember to clone to avoid in-place replacements!
                        sub_td.clone(),
                        # actions=sub_td["actions"],
                        env=self.env,
                        return_entropy=True,
                        return_sum_log_likelihood=False,
                    )
                    ll, entropy = out["log_likelihood"], out["entropy"]

                    # Compute the ratio of probabilities of new and old actions
                    ratio = torch.exp(ll.sum(dim=-1) - sub_td["logprobs"]).view(
                        -1, 1
                    )  # [batch, 1]

                    # Compute the advantage
                    value_pred = self.critic(sub_td)  # [batch, 1]
                    adv = previous_reward - value_pred.detach()

                    # Normalize advantage
                    if self.ppo_cfg["normalize_adv"]:
                        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                    # 调整维度
                    adv = adv.expand_as(sub_td["logprobs"]).reshape(-1, 1)

                    # Compute the surrogate loss
                    surrogate_loss = -torch.min(
                        ratio * adv,
                        torch.clamp(
                            ratio,
                            1 - self.ppo_cfg["clip_range"],
                            1 + self.ppo_cfg["clip_range"],
                        )
                        * adv,
                    ).mean()

                    # compute value function loss
                    value_loss = F.huber_loss(value_pred, previous_reward)

                    # compute total loss
                    loss = (
                        surrogate_loss
                        + self.ppo_cfg["vf_lambda"] * value_loss
                        - self.ppo_cfg["entropy_lambda"] * entropy.mean()
                    )

                    # perform manual optimization following the Lightning routine
                    # https://lightning.ai/docs/pytorch/stable/common/optimization.html

                    opt = self.optimizers()
                    opt.zero_grad()
                    self.manual_backward(loss)
                    if self.ppo_cfg["max_grad_norm"] is not None:
                        self.clip_gradients(
                            opt,
                            gradient_clip_val=self.ppo_cfg["max_grad_norm"],
                            gradient_clip_algorithm="norm",
                        )
                    opt.step()

            out.update(
                {
                    "loss": loss,
                    "surrogate_loss": surrogate_loss,
                    "value_loss": value_loss,
                    "entropy": entropy.mean(),
                }
            )

            # 添加PARCO特有的projection head损失计算
            # if self.use_projection_head and self.num_augment > 1:
            #     # 获取投影嵌入
            #     proj_embeds = self.projection_head(out["init_embeds"])

            #     # 对reward和log_likelihood进行unbatchify以进行损失计算
            #     reward = unbatchify(out["reward"], (1, self.num_augment))
            #     ll = unbatchify(out["log_likelihood"], (1, self.num_augment))

            #     # 计算invariance loss
            #     loss_inv = invariance_loss(proj_embeds, self.num_augment)

            #     # 计算solution symmetricity loss
            #     loss_ss = solution_symmetricity_loss(reward[..., None], ll, dim=1)

            #     # 无multi-start，所以problem symmetricity loss为0
            #     loss_ps = 0

            #     # 计算PARCO总损失
            #     parco_loss = loss_ps + self.beta * loss_ss + self.alpha * loss_inv

            #     # 更新总损失
            #     out["loss"] = out["loss"] + parco_loss

            #     # 添加PARCO相关指标到输出
            #     out.update({
            #         "parco_loss": parco_loss,
            #         "loss_ss": loss_ss,
            #         "loss_inv": loss_inv,
            #     })

            #     # 记录PARCO指标
            #     self.log("train/loss_ss", loss_ss, prog_bar=False)
            #     self.log("train/loss_inv", loss_inv, prog_bar=False)
            #     self.log("train/parco_loss", parco_loss, prog_bar=False)

        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}

    # 无法向父类添加参数，需要重写
    # def configure_optimizers(self):
    #     """Configure optimizers to include projection head parameters if needed"""
    #     # Start with parameters from policy and critic (from parent class)
    #     parameters = list(self.policy.parameters()) + list(self.critic.parameters())

    #     # Add projection head parameters if using it
    #     if self.use_projection_head:
    #         parameters += list(self.projection_head.parameters())

    #     return super().configure_optimizers(parameters)
