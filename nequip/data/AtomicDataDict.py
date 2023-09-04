"""nequip.data.jit: TorchScript functions for dealing with AtomicData.

These TorchScript functions operate on ``Dict[str, torch.Tensor]`` representations
of the ``AtomicData`` class which are produced by ``AtomicData.to_AtomicDataDict()``.

Authors: Albert Musaelian
"""
from typing import Dict, Any

import torch
import torch.jit

from e3nn import o3

# Make the keys available in this module
from ._keys import *  # noqa: F403, F401

# Also import the module to use in TorchScript, this is a hack to avoid bug:
# https://github.com/pytorch/pytorch/issues/52312
from . import _keys

# Define a type alias
Type = Dict[str, torch.Tensor]


def validate_keys(keys, graph_required=True):
    # Validate combinations
    if graph_required:
        if not (_keys.POSITIONS_KEY in keys and _keys.EDGE_INDEX_KEY in keys):
            raise KeyError("At least pos and edge_index must be supplied")
    if _keys.EDGE_CELL_SHIFT_KEY in keys and "cell" not in keys:
        raise ValueError("If `edge_cell_shift` given, `cell` must be given.")


_SPECIAL_IRREPS = [None]


def _fix_irreps_dict(d: Dict[str, Any]):
    # irreps를 하나씩 받다가 뭐가 잘못되면 고쳐주는 애인듯
    # 보통 받는 d를 살펴보면
    # {}
    # {'pos': 1x1o, 'edge_index': None, 'node_attrs': 2x0e, 'node_features': 2x0e}
    # {'pos': 1x1o, 'edge_index': None, 'node_attrs': 2x0e, 'node_features': 2x0e, 'edge_attrs': 1x0e+1x1o+1x2e}
    # {'pos': 1x1o, 'edge_index': None, 'node_attrs': 2x0e, 'node_features': 32x0e, 'edge_attrs': 1x0e+1x1o+1x2e, 'edge_embedding': 8x0e}
    # {'pos': 1x1o, 'edge_index': None, 'node_attrs': 2x0e, 'node_features': 16x0e, 'edge_attrs': 1x0e+1x1o+1x2e, 'edge_embedding': 8x0e}
    # {'pos': 1x1o, 'edge_index': None, 'node_attrs': 2x0e, 'node_features': 16x0e, 'edge_attrs': 1x0e+1x1o+1x2e, 'edge_embedding': 8x0e, 'atomic_energy': 1x0e}
    # 뭔가를 진짜 고치는 경우는 드문듯?
    return {k: (i if i in _SPECIAL_IRREPS else o3.Irreps(i)) for k, i in d.items()}


def _irreps_compatible(ir1: Dict[str, o3.Irreps], ir2: Dict[str, o3.Irreps]):
    # compatible 한지 확인함 맞으면 True를 아니면 False를 return 하는듯
    # ir1: {'pos': 1x1o, 'edge_index': None, 'node_attrs': 2x0e, 'node_features': 2x0e, 'edge_attrs': 1x0e+1x1o+1x2e}
    # ir2: {'pos': 1x1o, 'edge_index': None, 'node_attrs': 2x0e, 'node_features': 2x0e, 'edge_attrs': 1x0e+1x1o+1x2e}
    # 얘도 위에 irreps가 update되면서 점점 바뀜
    # 
    return all(ir1[k] == ir2[k] for k in ir1 if k in ir2)


@torch.jit.script
def with_edge_vectors(data: Type, with_lengths: bool = True) -> Type:
    # data는 dict 인 것 같음
    # key는   edge_index, 
    #         pos (atom마다 position)
    #         batch, 
    #         ptr: 이게 뭔지 모르겠음
    #         free_energy: 
    #         pbc(방향마다 boolean)
    #         edge_cell_shift: edge가 pbc의 어떤방향으로 어떻게 건너서 존재하는지
    #         cell: cell 에 대해서 9개 component matrix(3x3)
    #         r_max:
    #         atomtypes: atom별 atomtypes
    #         node_attrs: 1  0 << 이렇게 n atom에 대해 쓰여있는데 뭔지는 잘 모르겠음
    
    """
    {edge_index: Columns 1 to 20  0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   1   1   1   1   1
  1  49   4  16  35  30  25  58  23  54  41  13  42  38  58  30  56   0  55  48
    ...
    ...
    ...
Columns 801 to 814 62  62  62  62  62  62  63  63  63  63  63  63  63  63
 26  46  12  15  40  60   6   8  34   7  35  27  16  33

+ 
    """
    
        
    # 어떤 dictionary를 return 하는데 
    # key는 다음과 같음
    # edge_vectors (상대적 displacement vector)
    # edge_attrs(Columns 1 to 6 and 7 to 9): 뭔지 모름
    # edge_length: edge 길이
    
    """Compute the edge displacement vectors for a graph.

    If ``data.pos.requires_grad`` and/or ``data.cell.requires_grad``, this
    method will return edge vectors correctly connected in the autograd graph.

    

    
    Returns:
        Tensor [n_edges, 3] edge displacement vectors
    """
    if _keys.EDGE_VECTORS_KEY in data:
        if with_lengths and _keys.EDGE_LENGTH_KEY not in data:
            data[_keys.EDGE_LENGTH_KEY] = torch.linalg.norm(
                data[_keys.EDGE_VECTORS_KEY], dim=-1
            )
        return data
    else:
        # Build it dynamically
        # Note that this is
        # (1) backwardable, because everything (pos, cell, shifts)
        #     is Tensors.
        # (2) works on a Batch constructed from AtomicData
        pos = data[_keys.POSITIONS_KEY]
        edge_index = data[_keys.EDGE_INDEX_KEY]
        edge_vec = pos[edge_index[1]] - pos[edge_index[0]]
        if _keys.CELL_KEY in data:
            # ^ note that to save time we don't check that the edge_cell_shifts are trivial if no cell is provided; we just assume they are either not present or all zero.
            # -1 gives a batch dim no matter what
            cell = data[_keys.CELL_KEY].view(-1, 3, 3)
            edge_cell_shift = data[_keys.EDGE_CELL_SHIFT_KEY]
            if cell.shape[0] > 1:
                batch = data[_keys.BATCH_KEY]
                # Cell has a batch dimension
                # note the ASE cell vectors as rows convention
                edge_vec = edge_vec + torch.einsum(
                    "ni,nij->nj", edge_cell_shift, cell[batch[edge_index[0]]]
                )
                # TODO: is there a more efficient way to do the above without
                # creating an [n_edge] and [n_edge, 3, 3] tensor?
            else:
                # Cell has either no batch dimension, or a useless one,
                # so we can avoid creating the large intermediate cell tensor.
                # Note that we do NOT check that the batch array, if it is present,
                # is trivial — but this does need to be consistent.
                edge_vec = edge_vec + torch.einsum(
                    "ni,ij->nj",
                    edge_cell_shift,
                    cell.squeeze(0),  # remove batch dimension
                )
        data[_keys.EDGE_VECTORS_KEY] = edge_vec
        if with_lengths:
            data[_keys.EDGE_LENGTH_KEY] = torch.linalg.norm(edge_vec, dim=-1)
        return data


@torch.jit.script
def with_batch(data: Type) -> Type:
    """Get batch Tensor.

    If this AtomicDataPrimitive has no ``batch``, one of all zeros will be
    allocated and returned.
    """
    if _keys.BATCH_KEY in data:
        return data
    else:
        pos = data[_keys.POSITIONS_KEY]
        batch = torch.zeros(len(pos), dtype=torch.long, device=pos.device)
        data[_keys.BATCH_KEY] = batch
        return data
