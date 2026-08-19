import torch
from torch_geometric.data import Data
import random

from graphphysics.utils.nodetype import NodeType

device = "cuda" if torch.cuda.is_available() else "cpu"


STENT_IDX = {
  "S1": 1,
  "S2": 2,
  "S3": 3
}

STENT_FT = {
  "S1": [36, 0.35756],
  "S2": [24, 0.5363],
  "S3": [16, 0.805]
}


def build_features(graph: Data) -> Data:
    # idx = random.randint(0, graph.x.shape[0] - 1)
    # print(f"IDX: {idx}")
    # print(f"BEFORE: {graph.x[idx]}")
    inlet_lvlst = graph.x[:, 3]
    centerline_lvlst = graph.x[:, 4]
    stent_lvlst = torch.zeros(centerline_lvlst.shape[0], device=device)
    node_type = graph.x[:, 5]
    timestep = graph.x[:, 6]

    stent_index = 0
    stent_one_hot = torch.zeros((centerline_lvlst.shape[0], 4), device=device)
    stent_features = torch.zeros((centerline_lvlst.shape[0], 5), device=device)

    # if NodeType.ANEURYSM in torch.unique(node_type):
    #     # print("ANEURYSM NODE TYPE FOUND")
    #     # switch all aneurysm node types to normal
    #     node_type[node_type == NodeType.ANEURYSM] = NodeType.NORMAL

    if NodeType.STENT in torch.unique(node_type):
        stent_number = graph.id.split("_")[-1]
        # print(f"stent_num: {stent_number}")
        stent_lvlst = graph.x[:, 6]
        timestep = graph.x[:, 7]
        stent_index = STENT_IDX[stent_number]
        features = STENT_FT[stent_number]
        stent_features[:, 0] = features[0]
        stent_features[:, 1] = features[1]

    stent_one_hot[:, stent_index] = torch.ones(centerline_lvlst.shape[0], device=device)

    # print("UNIQUE", torch.unique(node_type))
    # print("center max", torch.max(centerline_lvlst))
    # print("inlet max", torch.max(inlet_lvlst))

    current_velocity = graph.x[:, 0:3]
    target_velocity = graph.y[:, 0:3]
    previous_velocity = torch.tensor(graph.previous_data["Vitesse"], device=device)

    acceleration = current_velocity - previous_velocity

    norm_next_acceleration = torch.norm(target_velocity, dim=1) - torch.norm(
        current_velocity, dim=1
    )
    not_inflow_mask = node_type != NodeType.INFLOW
    norm_next_acceleration[not_inflow_mask] = 0
    mean_next_accel = torch.ones(node_type.shape, device=device) * torch.mean(
        norm_next_acceleration
    )
    # print(f"CHECK: {torch.max(current_velocity[not_inflow_mask])} and {torch.min(current_velocity[not_inflow_mask])}")
    # print("accel and timestep", torch.mean(norm_next_acceleration), torch.unique(timestep))

    graph.x = torch.cat(
        (
            current_velocity,
            timestep.unsqueeze(1),
            acceleration,
            graph.pos,
            mean_next_accel.unsqueeze(1),
            inlet_lvlst.unsqueeze(1),
            centerline_lvlst.unsqueeze(1),
            stent_lvlst.unsqueeze(1),
            stent_one_hot,
            stent_features,
            node_type.to(device).unsqueeze(1),
        ),
        dim=1,
    )

    # print(f"AFTER: {graph.x[idx]}")
    return graph


def build_features_w_wss(graph: Data) -> Data:
    # print(f"\nBEFORE: {graph.x[6000]}")
    # print(graph)
    inlet_lvlst = graph.x[:, 6]
    centerline_lvlst = graph.x[:, 7]
    node_type = graph.x[:, 8]
    timestep = graph.x[:, 9]
    stent_one_hot = torch.zeros((centerline_lvlst.shape[0], 4), device=device)
    stent_index = 0
    stent_lvlst = torch.zeros(centerline_lvlst.shape[0], device=device)
    stent_features = torch.zeros((centerline_lvlst.shape[0], 5), device=device)

    if NodeType.STENT in torch.unique(node_type):
        stent_number = graph.id.split("_")[-1]
        # print(f"stent_num: {stent_number}")
        stent_lvlst = graph.x[:, 9]
        timestep = graph.x[:, 10]
        stent_index = STENT_IDX[stent_number]
        features = STENT_FT[stent_number]
        stent_features[:, 0] = features[0]
        stent_features[:, 1] = features[1]

    stent_one_hot[:, stent_index] = torch.ones(centerline_lvlst.shape[0], device=device)

    # if NodeType.ANEURYSM in torch.unique(node_type):
    #     # print("ANEURYSM NODE TYPE FOUND")
    #     # switch all aneurysm node types to normal
    #     node_type[node_type == NodeType.ANEURYSM] = NodeType.NORMAL

    # print("UNIQUE", torch.unique(node_type))

    current_velocity = graph.x[:, 0:3]
    current_wss = graph.x[:, 3:6]
    target_velocity = graph.y[:, 0:3]
    previous_velocity = torch.tensor(graph.previous_data["Vitesse"], device=device)

    acceleration = current_velocity - previous_velocity

    norm_next_acceleration = torch.norm(target_velocity, dim=1) - torch.norm(
        current_velocity, dim=1
    )
    not_inflow_mask = node_type != NodeType.INFLOW
    norm_next_acceleration[not_inflow_mask] = 0
    mean_next_accel = torch.ones(node_type.shape, device=device) * torch.mean(
        norm_next_acceleration
    )

    graph.x = torch.cat(
        (
            current_velocity,
            current_wss,
            timestep.unsqueeze(1),
            acceleration,
            graph.pos,
            mean_next_accel.unsqueeze(1),
            inlet_lvlst.unsqueeze(1),
            centerline_lvlst.unsqueeze(1),
            stent_lvlst.unsqueeze(1),
            stent_one_hot,
            stent_features,
            node_type.to(device).unsqueeze(1),
        ),
        dim=1,
    )
    # print(f"AFTER: {graph.x[6000]}")

    return graph
