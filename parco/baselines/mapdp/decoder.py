from typing import Tuple

import torch.nn as nn

from rl4co.models.zoo.am.decoder import AttentionModelDecoder
from rl4co.utils.ops import unbatchify
from rl4co.utils.pylogger import get_pylogger
from tensordict import TensorDict
from torch import Tensor

from .env_embeddings import env_context_embedding, env_dynamic_embedding

log = get_pylogger(__name__)


class MAPDPDecoder(AttentionModelDecoder):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        env_name: str = "hcvrp",
        context_embedding: nn.Module = None,
        context_embedding_kwargs: dict = {},
        dynamic_embedding: nn.Module = None,
        use_graph_context: bool = False,
        use_pos_token: bool = False,
        **kwargs,
    ):
        self.use_pos_token = use_pos_token
        context_embedding_kwargs["embed_dim"] = embed_dim  # replace
        if context_embedding is None:
            context_embedding = env_context_embedding(env_name, context_embedding_kwargs)

        if dynamic_embedding is None:
            dynamic_embedding = env_dynamic_embedding(env_name, {"embed_dim": embed_dim})

        if use_graph_context:
            raise ValueError("MAPDP does not use graph context")

        super(MAPDPDecoder, self).__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            env_name=env_name,
            context_embedding=context_embedding,
            dynamic_embedding=dynamic_embedding,
            use_graph_context=use_graph_context,
            **kwargs,
        )

    def forward(
        self,
        td: TensorDict,
        cached,
        num_starts: int = 0,
        do_unbatchify: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Compute the logits of the next actions given the current state

        Args:
            cache: Precomputed embeddings
            td: TensorDict with the current environment state
            num_starts: Number of starts for the multi-start decoding
        """

        # i.e. during sampling, operate only once during all steps
        if num_starts > 1 and do_unbatchify:
            td = unbatchify(td, num_starts)
            td = td.contiguous().view(-1)

        glimpse_q = self._compute_q(cached, td)
        glimpse_k, glimpse_v, logit_k = self._compute_kvl(cached, td)

        # Masking: 1 means available, 0 means not available
        mask = td["action_mask"]

        # After pass communication layer reshape glimpse_q [B*S, m, N] -> [B, S*m, N] for efficient pointer attiention
        if num_starts > 1:
            batch_size = glimpse_k.shape[0]
            glimpse_q = glimpse_q.reshape(batch_size, -1, self.embed_dim)
            mask = mask.reshape(batch_size, glimpse_q.shape[1], -1)

        # Compute logits
        logits = self.pointer(glimpse_q, glimpse_k, glimpse_v, logit_k, mask)

        # Now we need to reshape the logits and mask to [B*S,N,...] is num_starts > 1 without dynamic embeddings
        # note that rearranging order is important here
        # TODO: check this
        # if num_starts > 1:
        #     logits = rearrange(logits, "b s l -> (s b) l", s=num_starts)
        #     mask = rearrange(mask, "b s l -> (s b) l", s=num_starts)
        if num_starts > 1:
            logits = logits.reshape(batch_size * num_starts, -1, logits.shape[-1])
            # print("final logits shape", logits.shape)
            # TODO: useless?
            mask = mask.reshape(batch_size * num_starts, -1, mask.shape[-1])
        return logits, mask

    def pre_decoder_hook(
        self, td, env, embeddings, num_starts: int = 0, num_samples: int = 1
    ):
        """Precompute the embeddings cache before the decoder is called"""
        cached = self._precompute_cache(embeddings, num_starts=num_starts)

        # when we do multi-sampling, only node embeddings are repeated
        if num_samples > 1:
            cached.node_embeddings = cached.node_embeddings.repeat_interleave(
                num_samples, dim=0
            )

        return td, env, cached
