import abc

from typing import Tuple

import torch

from rl4co.envs import RL4COEnvBase
from rl4co.utils.decoding import process_logits
from rl4co.utils.ops import batchify, gather_by_index, unbatchify, unbatchify_and_gather
from rl4co.utils.pylogger import get_pylogger
from tensordict.tensordict import TensorDict

log = get_pylogger(__name__)


def mapdp_get_decoding_strategy(decoding_strategy, **config):
    strategy_registry = {
        "greedy": Greedy,
        "sampling": Sampling,
        "evaluate": Evaluate,
    }

    if decoding_strategy not in strategy_registry:
        log.warning(
            f"Unknown decode type '{decoding_strategy}'. Available decode types: {strategy_registry.keys()}. Defaulting to Sampling."
        )

    if "multistart" in decoding_strategy:
        raise ValueError("Multistart is not supported for multi-agent decoding")

    return strategy_registry.get(decoding_strategy, Sampling)(**config)


class MAPDPDecodingStrategy(metaclass=abc.ABCMeta):
    name = "base"

    def __init__(
        self,
        num_agents: int,
        agent_handler=None,  # Agent handler
        use_init_logp: bool = True,  # Return initial logp for actions even with conflicts
        mask_handled: bool = False,  # Mask out handled actions (make logprobs 0)
        replacement_value_key: str = "current_node",  # When stopping arises (conflict or POS token), replace the value of this key
        use_pos_token: bool = False,  # Add a POS (pause-of-sequence) action
        dead_lock_checking: bool = False,  # Check for deadlock
        temperature: float = 1.0,
        top_p: float = 0.0,
        top_k: int = 0,
        tanh_clipping: float = 10.0,
        multistart: bool = False,
        multisample: bool = False,
        num_samples: int = 1,
        select_best: bool = False,
        store_all_logp: bool = False,
    ) -> None:
        if mask_handled and agent_handler is None:
            raise ValueError(
                "mask_handled is only supported when agent_handler is not None for now"
            )

        if store_all_logp and mask_handled:
            raise ValueError("store_all_logp is not supported when mask_handled is True")

        if mask_handled and use_init_logp:
            raise ValueError(
                "We should not mask out the initial action logp, rather the final action logp"
            )

        self.use_init_logp = use_init_logp
        self.mask_handled = mask_handled
        self.store_all_logp = store_all_logp
        self.use_pos_token = use_pos_token
        self.num_agents = num_agents
        self.agent_handler = agent_handler
        self.replacement_value_key = replacement_value_key
        self.dead_lock_checking = dead_lock_checking

        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.tanh_clipping = tanh_clipping
        if multistart:
            raise ValueError("Multistart is not supported for multi-agent decoding")
        self.multistart = multistart
        self.multisample = multisample
        self.num_samples = num_samples
        if self.num_samples > 1:
            self.multisample = True
        self.select_best = select_best

        # initialize buffers
        self.actions = []
        self.logprobs = []
        self.handling_masks = []
        self.halting_ratios = []
        self.pos_ratios = []
        self.iter_count = 0

    @abc.abstractmethod
    def _step(
        self,
        logprobs: torch.Tensor,
        mask: torch.Tensor,
        td: TensorDict,
        action: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
        raise NotImplementedError("Must be implemented by subclass")

    def pre_decoder_hook(
        self, td: TensorDict, env: RL4COEnvBase, action: torch.Tensor = None
    ):
        """Pre decoding hook. This method is called before the main decoding operation."""

        if self.num_samples >= 1:
            # Expand td to batch_size * num_samples
            td = batchify(td, self.num_samples)

        return td, env, self.num_samples

    def post_decoder_hook(
        self, td: TensorDict, env: RL4COEnvBase
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict, RL4COEnvBase]:
        """ "
        Size depends on whether we store all log p or not. By default, we don't
        Returns:
            logprobs: [B, m, L]
            actions: [B, m, L]
        """
        assert (
            len(self.logprobs) > 0
        ), "No logprobs were collected because all environments were done. Check your initial state"
        # [B, m, L] (or is it?)
        logprobs = torch.stack(self.logprobs, -1)
        actions = torch.stack(self.actions, -1)

        if len(self.handling_masks) > 0:
            if self.handling_masks[0] is not None:
                torch.stack(self.handling_masks, 2)
            else:
                pass
        else:
            pass

        halting_ratios = (
            torch.stack(self.halting_ratios, 0) if len(self.halting_ratios) > 0 else 0
        )

        pos_ratios = torch.stack(self.pos_ratios, 0) if len(self.pos_ratios) > 0 else 0

        if self.num_samples > 0 and self.select_best:
            logprobs, actions, td, env = self._select_best(logprobs, actions, td, env)

        return logprobs, actions, td, env, halting_ratios, pos_ratios

    def step(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        td: TensorDict = None,
        **kwargs,
    ) -> TensorDict:
        self.iter_count += 1

        logprobs = process_logits(
            logits,
            mask,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            tanh_clipping=self.tanh_clipping,
        )

        logprobs, actions, td = self._step(logprobs, mask, td, **kwargs)
        actions_init = actions.clone()

        if self.agent_handler is not None:
            # Solve conflicts via agent handler
            replacement_value = td[
                self.replacement_value_key
            ]  # replace with previous node

            actions, handling_mask, halting_ratio = self.agent_handler(
                actions, replacement_value, td, probs=logprobs.clone()
            )
            self.handling_masks.append(handling_mask)
            self.halting_ratios.append(halting_ratio)

        # for others
        if not self.store_all_logp:
            if self.use_pos_token:
                # Use the initial actions for gathering the logprobs, i.e. before processing the deadlock
                # and replace the POS token with the current node.
                actions_gather = actions_init
            else:
                actions_gather = actions_init if self.use_init_logp else actions
            # logprobs: [B, m, N], actions_cur: [B, m]
            # transform logprobs to [B, m]

            logprobs = gather_by_index(logprobs, actions_gather, dim=-1)

            # We do this after gathering the logprobs
            if self.mask_handled:
                logprobs.masked_fill_(handling_mask, 0)

        td.set("action", actions)
        self.actions.append(actions)
        self.logprobs.append(logprobs)

        # Check if there exists a deadlock, only for POS token
        if self.use_pos_token & self.dead_lock_checking & len(self.actions) > 1:
            all_same_actions_flag = (self.actions[-1] == self.actions[-2]).all(dim=-1)
            not_done_flag = ~td["done"]
            assert ~(
                all_same_actions_flag & not_done_flag
            ).any(), "Actions are the same and not done"

        return td

    @staticmethod
    def greedy(logprobs, mask=None):
        """Select the action with the highest probability."""
        selected = logprobs.argmax(dim=-1)  # [B, m, N] -> [B, m]
        if mask is not None:  # [B, m, N]
            assert (
                not (~mask).gather(-1, selected.unsqueeze(-1)).data.any()
            ), "infeasible action selected"
        return selected

    @staticmethod
    def sampling(logprobs, mask=None):
        """Sample an action with a multinomial distribution given by the log probabilities."""

        distribution = torch.distributions.Categorical(logits=logprobs)
        selected = distribution.sample()  # samples [B, m, N] -> [B, m]

        if mask is not None:
            # checking for bad values sampling; but is this needed?
            while (~mask).gather(-1, selected.unsqueeze(-1)).data.any():
                log.info("Sampled bad values, resampling!")
                # selected = probs.multinomial(1).squeeze(1)
                selected = distribution.sample()
            assert (
                not (~mask).gather(-1, selected.unsqueeze(-1)).data.any()
            ), "infeasible action selected"
        return selected

    def _select_best(self, logprobs, actions, td: TensorDict, env: RL4COEnvBase):
        # TODO: check
        rewards = env.get_reward(td, actions)
        _, max_idxs = unbatchify(rewards, self.num_samples).max(dim=-1)

        actions = unbatchify_and_gather(actions, max_idxs, self.num_samples)
        logprobs = unbatchify_and_gather(logprobs, max_idxs, self.num_samples)
        td = unbatchify_and_gather(td, max_idxs, self.num_samples)

        return logprobs, actions, td, env


class Greedy(MAPDPDecodingStrategy):
    name = "greedy"

    def _step(
        self, logprobs: torch.Tensor, mask: torch.Tensor, td: TensorDict, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
        """Select the action with the highest log probability"""
        selected = self.greedy(logprobs, mask)
        return logprobs, selected, td


class Sampling(MAPDPDecodingStrategy):
    name = "sampling"

    def _step(
        self, logprobs: torch.Tensor, mask: torch.Tensor, td: TensorDict, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
        """Sample an action with a multinomial distribution given by the log probabilities."""
        selected = self.sampling(logprobs, mask)
        return logprobs, selected, td


class Evaluate(MAPDPDecodingStrategy):
    name = "evaluate"

    def _step(
        self,
        logprobs: torch.Tensor,
        mask: torch.Tensor,
        td: TensorDict,
        action: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
        """The action is provided externally, so we just return the action"""
        selected = action
        return logprobs, selected, td
