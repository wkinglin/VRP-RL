from typing import Tuple, Union

import torch.nn as nn

from tensordict import TensorDict
from torch import Tensor

from parco.models.nn.transformer import Normalization, TransformerBlock

from .env_embeddings import env_init_embedding


class MAPDPEncoder(nn.Module):
    def __init__(
        self,
        env_name: str = "omdcpdp",
        num_heads: int = 8,
        embed_dim: int = 128,
        num_layers: int = 3,
        normalization: str = "instance",
        use_final_norm: bool = False,
        init_embedding: nn.Module = None,
        init_embedding_kwargs: dict = {},
        norm_after: bool = False,
        **transformer_kwargs,
    ):
        super(MAPDPEncoder, self).__init__()

        self.env_name = env_name
        init_embedding_kwargs["embed_dim"] = embed_dim
        self.init_embedding = (
            init_embedding
            if init_embedding is not None
            else env_init_embedding(self.env_name, init_embedding_kwargs)
        )

        self.layers = nn.Sequential(
            *(
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    normalization=normalization,
                    norm_after=norm_after,
                    **transformer_kwargs,
                )
                for _ in range(num_layers)
            )
        )

        self.norm = Normalization(embed_dim, normalization) if use_final_norm else None

    def forward(
        self, td: TensorDict, mask: Union[Tensor, None] = None
    ) -> Tuple[Tensor, Tensor]:
        # Transfer to embedding space
        init_h = self.init_embedding(td)  # [B, N, H]

        # Process embedding
        h = init_h
        for layer in self.layers:
            h = layer(h, mask)

        # https://github.com/meta-llama/llama/blob/8fac8befd776bc03242fe7bc2236cdb41b6c609c/llama/model.py#L493
        if self.norm is not None:
            h = self.norm(h)

        # Return latent representation and initial embedding
        # [B, N, H]
        return h, init_h
