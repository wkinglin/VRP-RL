import torch
import torch.nn as nn

from rl4co.utils.ops import gather_by_index

from parco.models.env_embeddings.communication import BaseMultiAgentContextEmbedding


class OMDCPDPInitEmbedding(nn.Module):
    """Encoder for the initial state of the OMDCPDP environment with MAPDP paper
    MAPAP employs a paired initial node embedding for pickup-delivery problems
    Encode the initial state of the environment into a fixed-size vector
    Features:
        - depot: location
        - pickup: location, corresponding delivery location and demand
        - delivery: location
    """

    def __init__(self, embed_dim: int):
        super(OMDCPDPInitEmbedding, self).__init__()
        node_dim = 2

        # Original embeddings for all nodes
        self.init_embed_general = nn.Linear(node_dim, embed_dim)

        # Depot embeddings
        self.init_embed_depot = nn.Linear(embed_dim, embed_dim)

        # Pickup embeddings (pickup and paired delivery)
        self.init_embed_pick = nn.Linear(embed_dim * 2, embed_dim)

        # Delivery embeddings
        self.init_embed_delivery = nn.Linear(embed_dim, embed_dim)

    def forward(self, td):
        num_agents = td["current_node"].size(-1)
        num_cities = td["locs"].shape[-2] - num_agents
        num_pickup = num_agents + int(num_cities / 2)

        locs = td["locs"]

        # Init embeddings for all nodes
        general_embed = self.init_embed_general(locs)
        depot_embed_init, pickup_embed_init, delivery_embed_init = (
            general_embed[..., :num_agents, :],
            general_embed[..., num_agents:num_pickup, :],
            general_embed[..., num_pickup:, :],
        )

        # Depot embeddings
        depot_embed = self.init_embed_depot(depot_embed_init)

        # Pickup embeddings
        pickup_embed = self.init_embed_pick(
            torch.cat([pickup_embed_init, delivery_embed_init], dim=-1)
        )

        # Delivery embeddings
        delivery_embed = self.init_embed_delivery(delivery_embed_init)

        return torch.cat([depot_embed, pickup_embed, delivery_embed], dim=-2)


class OMDCPDPContextEmbedding(BaseMultiAgentContextEmbedding):
    """Context embedding for OMDCPDP"""

    def __init__(
        self,
        embed_dim,
        num_agents,
        agent_feat_dim=2,
        global_feat_dim=2,
        linear_bias=False,
        use_communication=True,
        use_final_norm=False,
        num_communication_layers=1,
        **communication_kwargs,
    ):
        super(BaseMultiAgentContextEmbedding, self).__init__()
        self.embed_dim = embed_dim

        # Feature projection
        self.proj_agent_feats = nn.Linear(agent_feat_dim, embed_dim, bias=linear_bias)
        self.proj_communication_feats = nn.Linear(
            agent_feat_dim * num_agents, embed_dim, bias=linear_bias
        )
        self.proj_global_feats = nn.Linear(global_feat_dim, embed_dim, bias=linear_bias)
        self.project_context = nn.Linear(embed_dim * 3, embed_dim, bias=linear_bias)

    def _agent_state_embedding(self, embeddings, td, num_agents, num_cities):
        context_feats = torch.cat(
            [
                td["current_length"][..., None],  # cost
                td["num_orders"][..., None].float(),  # capacity
            ],
            dim=-1,
        )  # [B, M, 2]
        return self.proj_agent_feats(context_feats)  # [B, M, 2] -> [B, M, hdim]

    def _communication_embedding(self, embeddings, td, num_agents, num_cities):
        context_feats = torch.cat(
            [
                td["current_length"][..., None],  # cost
                td["num_orders"][..., None].float(),  # capacity
            ],
            dim=-1,
        )  # [B, M, 2]
        context_feats = context_feats.flatten(start_dim=1)  # [B, 2M]
        communication_embed = self.proj_communication_feats(
            context_feats
        )  # [B, 2M] -> [B, hdim]

        # Repeat the communication embedding for each agent
        return communication_embed[..., None, :].repeat(1, num_agents, 1)  # [B, M, hdim]

    def forward(self, embeddings, td):
        # Collect embeddings
        num_agents = td["action_mask"].shape[-2]
        num_cities = td["locs"].shape[-2] - num_agents

        # Current node embeddings
        cur_node_embedding = gather_by_index(
            embeddings, td["current_node"]
        )  # [B, M, hdim]

        # Agent state embeddings
        agent_state_embed = self._agent_state_embedding(
            embeddings, td, num_agents=num_agents, num_cities=num_cities
        )  # [B, M, hdim]

        # Communication embeddings
        communication_embed = self._communication_embedding(
            embeddings, td, num_agents=num_agents, num_cities=num_cities
        )  # [B, M, hdim]

        context_embed = torch.cat(
            [cur_node_embedding, agent_state_embed, communication_embed], dim=-1
        )

        # [B, M, 3 * hdim] -> [B, M, hdim]
        context_embed = self.project_context(context_embed)

        return context_embed
