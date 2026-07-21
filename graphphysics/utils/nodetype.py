import enum
import torch


class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9


GLOBAL_ATTENTION_NODE = NodeType.WALL_BOUNDARY


def build_mask_from_nodetypes(nodetype_tensor, include_nodetypes: list) -> torch.Tensor:
    """
    Build a mask from a list of node types.

    Args:
        nodetype_tensor (torch.Tensor): A tensor containing the graph node types.
        include_nodetypes (list): List of node types to include in the mask.

    Returns:
        torch.Tensor: A boolean mask where True indicates the presence of the specified node types.
    """
    mask = torch.zeros_like(nodetype_tensor, dtype=torch.bool)
    for node_type in include_nodetypes:
        mask = torch.logical_or(mask, nodetype_tensor == node_type)
    return mask
