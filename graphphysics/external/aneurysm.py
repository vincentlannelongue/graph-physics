import torch
from torch_geometric.data import Data

from graphphysics.utils.nodetype import NodeType

device = "cuda" if torch.cuda.is_available() else "cpu"


def build_features(graph: Data) -> Data:
    # print(f"BEFORE: {graph.x[6000]}")
    inlet_lvlst = graph.x[:, 4]
    node_type = graph.x[:, 4]
    timestep = graph.x[:, 5]
    # print("UNIQUE", torch.unique(node_type))

    current_velocity = graph.x[:, 0:3]
    target_velocity = graph.y[:, 0:3]
    previous_velocity = torch.tensor(graph.previous_data["Vitesse"], device=device)

    acceleration = current_velocity - previous_velocity
    next_acceleration = target_velocity - current_velocity

    not_inflow_mask = node_type != NodeType.INFLOW
    next_acceleration[not_inflow_mask] = 0
    next_acceleration_unique = next_acceleration.unique()

    mean_next_accel = torch.ones(node_type.shape, device=device) * torch.mean(
        next_acceleration_unique
    )
    min_next_accel = torch.ones(node_type.shape, device=device) * torch.min(
        next_acceleration_unique
    )
    max_next_accel = torch.ones(node_type.shape, device=device) * torch.max(
        next_acceleration_unique
    )

    graph.x = torch.cat(
        (
            current_velocity,
            timestep.unsqueeze(1),
            acceleration,
            graph.pos,
            mean_next_accel.unsqueeze(1),
            min_next_accel.unsqueeze(1),
            max_next_accel.unsqueeze(1),
            # inlet_lvlst.unsqueeze(1),
            node_type.to(device).unsqueeze(1),
        ),
        dim=1,
    )

    # print(f"AFTER: {graph.x[6000]}")
    return graph


def build_features_w_wss(graph: Data) -> Data:
    pass
#     print(f"BEFORE: {graph.x[6000]}")

#     inlet_lvlst = graph.x[:, 6]
#     node_type = graph.x[:, 7]
#     timestep = graph.x[:, 8]

#     current_velocity = graph.x[:, 0:3]
#     current_wss = graph.x[:, 3:6]
#     target_velocity = graph.y[:, 0:3]
#     previous_velocity = torch.tensor(graph.previous_data["Vitesse"], device=device)

#     acceleration = current_velocity - previous_velocity
#     next_acceleration = target_velocity - current_velocity

#     not_inflow_mask = node_type != NodeType.INFLOW
#     next_acceleration[not_inflow_mask] = 0
#     next_acceleration_unique = next_acceleration.unique()

#     mean_next_accel = torch.ones(node_type.shape, device=device) * torch.mean(
#         next_acceleration_unique
#     )
#     min_next_accel = torch.ones(node_type.shape, device=device) * torch.min(
#         next_acceleration_unique
#     )
#     max_next_accel = torch.ones(node_type.shape, device=device) * torch.max(
#         next_acceleration_unique
#     )
#     velocity_norm = torch.norm(current_velocity, dim=1)
#     velocity_norm = velocity_norm.unsqueeze(1)

#     graph.x = torch.cat(
#         (
#             current_velocity,
#             current_wss,
#             timestep.unsqueeze(1),
#             acceleration,
#             graph.pos,
#             mean_next_accel.unsqueeze(1),
#             min_next_accel.unsqueeze(1),
#             max_next_accel.unsqueeze(1),
#             velocity_norm,
#             inlet_lvlst.unsqueeze(1),
#             node_type.to(device).unsqueeze(1),
#         ),
#         dim=1,
#     )
#     print(f"AFTER: {graph.x[6000]}")

#     return graph
