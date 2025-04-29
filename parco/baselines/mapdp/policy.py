import torch.nn as nn

from rl4co.utils.pylogger import get_pylogger
from tensordict import TensorDict

from parco.models.agent_handlers import get_agent_handler
from parco.models.utils import get_log_likelihood

from .decoder import MAPDPDecoder
from .decoding_strategies import MAPDPDecodingStrategy, mapdp_get_decoding_strategy
from .encoder import MAPDPEncoder

log = get_pylogger(__name__)


DEFAULTS_CONTEXT_KWARGS = {
    "use_communication": True,
    "num_communication_layers": 1,
    "normalization": "instance",
    "norm_after": False,
    "use_final_norm": False,
}


class MAPDPPolicy(nn.Module):
    """Policy for MAPDP model"""

    def __init__(
        self,
        encoder=None,
        decoder=None,
        embed_dim: int = 128,
        num_encoder_layers: int = 3,
        num_heads: int = 8,
        init_embedding: nn.Module = None,
        init_embedding_kwargs: dict = {},
        context_embedding: nn.Module = None,
        context_embedding_kwargs: dict = {},
        normalization: str = "instance",
        use_final_norm: bool = False,  # If True, normalize like in Llama
        norm_after: bool = False,
        env_name: str = "omdcpdp",
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "greedy",
        agent_handler=None,  # Agent handler
        agent_handler_kwargs: dict = {},  # Agent handler kwargs
        use_init_logp: bool = True,  # Return initial logp for actions even with conflicts
        mask_handled: bool = False,  # Mask out handled actions (make logprobs 0)
        replacement_value_key: str = "current_node",  # When stopping arises (conflict or POS token), replace the value of this key
        use_pos_token: bool = False,  # Add a POS (pause-of-sequence) action
        trainable_pos_token: bool = True,  # If true, then the pos token is trainable
        parallel_gated_kwargs: dict = None,  # ParallelGatedMLP kwargs
    ):
        super(MAPDPPolicy, self).__init__()

        if agent_handler is not None:
            if isinstance(agent_handler, str):
                agent_handler = get_agent_handler(agent_handler, **agent_handler_kwargs)
        self.agent_handler = agent_handler

        # If key is not provided, use default context kwargs
        context_embedding_kwargs = {
            **DEFAULTS_CONTEXT_KWARGS,
            **context_embedding_kwargs,
        }

        self.env_name = env_name

        # Encoder
        if encoder is None:
            log.info("Initializing default PARCOEncoder")
            self.encoder = MAPDPEncoder(
                env_name=self.env_name,
                num_heads=num_heads,
                embed_dim=embed_dim,
                num_layers=num_encoder_layers,
                normalization=normalization,
                init_embedding=init_embedding,
                init_embedding_kwargs=init_embedding_kwargs,
                use_final_norm=use_final_norm,
                norm_after=norm_after,
                parallel_gated_kwargs=parallel_gated_kwargs,
            )
        else:
            log.warning("Using custom encoder")
            self.encoder = encoder

        # Decoder
        if decoder is None:
            log.info("Initializing default PARCODecoder")
            self.decoder = MAPDPDecoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                context_embedding=context_embedding,
                context_embedding_kwargs=context_embedding_kwargs,
                env_name=self.env_name,
                use_pos_token=use_pos_token,
            )
        else:
            log.warning("Using custom decoder")
            self.decoder = decoder

        self.train_decode_type = train_decode_type
        self.val_decode_type = val_decode_type
        self.test_decode_type = test_decode_type

        # Multi-agent handling
        self.replacement_value_key = replacement_value_key
        self.use_pos_token = use_pos_token
        self.mask_handled = mask_handled
        self.use_init_logp = use_init_logp

    def forward(
        self,
        td: TensorDict,
        env,
        phase: str = "train",
        calc_reward: bool = True,
        return_actions: bool = True,
        return_sum_log_likelihood: bool = True,
        actions=None,
        max_steps=1_000_000,
        return_init_embeds: bool = True,
        **decoding_kwargs,
    ) -> dict:
        # Encoder: get encoder output and initial embeddings from initial state
        hidden, init_embeds = self.encoder(td)

        # Get decode type depending on phase and whether actions are passed for evaluation
        decode_type = decoding_kwargs.pop("decode_type", None)
        if actions is not None:
            decode_type = "evaluate"
        elif decode_type is None:
            decode_type = getattr(self, f"{phase}_decode_type")

        # When decode_type is sampling, we need to know the number of samples
        num_samples = decoding_kwargs.pop("num_samples", 1)
        if "sampling" not in decode_type:
            num_samples = 1

        # [B, m, N]
        num_agents = td["action_mask"].shape[-2]

        # Setup decoding strategy
        # we pop arguments that are not part of the decoding strategy
        decode_strategy: MAPDPDecodingStrategy = mapdp_get_decoding_strategy(
            decode_type,
            num_agents=num_agents,
            agent_handler=self.agent_handler,
            use_init_logp=self.use_init_logp,
            mask_handled=self.mask_handled,
            replacement_value_key=self.replacement_value_key,
            num_samples=num_samples,
            **decoding_kwargs,
        )

        # Pre-decoding hook: used for the initial step(s) of the decoding strategy
        td, env, num_samples = decode_strategy.pre_decoder_hook(td, env)
        num_embeds_repeat = num_samples * num_agents

        # Additionally call a decoder hook if needed before main decoding
        td, env, hidden = self.decoder.pre_decoder_hook(
            td, env, hidden, num_embeds_repeat, num_samples=num_samples
        )

        # We use unbatchify if num_samples > 1
        if num_samples > 1:
            do_unbatchify = True
        else:
            do_unbatchify = False

        # Main decoding: loop until all sequences are done
        step = 0
        while not td["done"].all():
            # We only need to proceed once when decoder forwarding.
            if step > 1:
                do_unbatchify = False

            logits, mask = self.decoder(td, hidden, num_samples, do_unbatchify)
            td = decode_strategy.step(
                logits,
                mask,
                td,
                action=actions[..., step] if actions is not None else None,
            )
            td = env.step(td)["next"]  # do not save the state
            step += 1
            if step > max_steps:
                log.error(
                    f"Exceeded maximum number of steps ({max_steps}) during decoding"
                )
                break

        # Post-decoding hook: used for the final step(s) of the decoding strategy
        (
            logprobs,
            actions,
            td,
            env,
            halting_ratio,
            pos_ratio,
        ) = decode_strategy.post_decoder_hook(td, env)

        # Output dictionary construction
        if calc_reward:
            td.set("reward", env.get_reward(td, actions))

        outdict = {
            "reward": td["reward"],
            "log_likelihood": get_log_likelihood(
                logprobs, actions, td.get("mask", None), return_sum_log_likelihood
            ),
            "halting_ratio": halting_ratio,
            "pos_ratio": pos_ratio,
        }

        if return_actions:
            outdict["actions"] = actions

        if return_init_embeds:  # for SymNCO
            outdict["init_embeds"] = init_embeds

        return outdict
