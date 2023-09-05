from typing import Optional
import logging

from e3nn import o3

from nequip.data import AtomicDataDict, AtomicDataset
from nequip.nn import (
    SequentialGraphNetwork,
    AtomwiseLinear,
    AtomwiseReduce,
    ConvNetLayer,
)
from nequip.nn.embedding import (
    OneHotAtomEncoding,
    RadialBasisEdgeEncoding,
    SphericalHarmonicEdgeAttrs,
)

from . import builder_utils
import torch

def SimpleIrrepsConfig(config, prefix: Optional[str] = None):
    """Builder that pre-processes options to allow "simple" configuration of irreps."""

    # We allow some simpler parameters to be provided, but if they are,
    # they have to be correct and not overridden
    simple_irreps_keys = ["l_max", "parity", "num_features"]
    real_irreps_keys = [
        "chemical_embedding_irreps_out",
        "feature_irreps_hidden",
        "irreps_edge_sh",
        "conv_to_output_hidden_irreps_out",
    ]

    prefix = "" if prefix is None else f"{prefix}_"

    has_simple: bool = any(
        (f"{prefix}{k}" in config) or (k in config) for k in simple_irreps_keys
    )
    has_full: bool = any(
        (f"{prefix}{k}" in config) or (k in config) for k in real_irreps_keys
    )
    assert has_simple or has_full

    update = {}
    if has_simple:
        # nothing to do if not
        lmax = config.get(f"{prefix}l_max", config["l_max"])
        parity = config.get(f"{prefix}parity", config["parity"])
        num_features = config.get(f"{prefix}num_features", config["num_features"])
        update[f"{prefix}chemical_embedding_irreps_out"] = repr(
            o3.Irreps([(num_features, (0, 1))])  # n scalars
        )
        update[f"{prefix}irreps_edge_sh"] = repr(
            o3.Irreps.spherical_harmonics(lmax=lmax, p=-1 if parity else 1)
        )
        update[f"{prefix}feature_irreps_hidden"] = repr(
            o3.Irreps(
                [
                    (num_features, (l, p))
                    for p in ((1, -1) if parity else (1,))
                    for l in range(lmax + 1)
                ]
            )
        )
        update[f"{prefix}conv_to_output_hidden_irreps_out"] = repr(
            # num_features // 2  scalars
            o3.Irreps([(max(1, num_features // 2), (0, 1))])
        )

    # check update is consistant with config
    # (this is necessary since it is not possible
    #  to delete keys from config, so instead of
    #  making simple and full styles mutually
    #  exclusive, we just insist that if full
    #  and simple are provided, full must be
    #  consistant with simple)
    for k, v in update.items():
        if k in config:
            assert (
                config[k] == v
            ), f"For key {k}, the full irreps options had value `{config[k]}` inconsistant with the value derived from the simple irreps options `{v}`"
        config[k] = v


def EnergyModel(
    config, initialize: bool, dataset: Optional[AtomicDataset] = None
) -> SequentialGraphNetwork:
    """Base default energy model archetecture.

    For minimal and full configuration option listings, see ``minimal.yaml`` and ``example.yaml``.
    """
    logging.debug("Start building the network model")

    builder_utils.add_avg_num_neighbors(
        config=config, initialize=initialize, dataset=dataset
    )

    num_layers = config.get("num_layers", 3)

    layers = {
        # -- Encode --
        "one_hot": OneHotAtomEncoding,
        "spharm_edges": SphericalHarmonicEdgeAttrs,
        "radial_basis": RadialBasisEdgeEncoding,
        # -- Embed features --
        "chemical_embedding": AtomwiseLinear,
    }
    ################################################################
    ################################################################
    ## atomic envrionment descision tree로 설정
    edge_index_TCSM = data["edge_index"]
    atom_types_TCSM= data["atom_types"]
    
    count_TCSM = torch.bincount(edge_index_TCSM.reshape(-1)) # check how many edges the atom has.
                                                                  # for 5nm, the 10 would be good
    criteron_count = count_TCSM>10
    print(criterion_count)
    
    
    
        

    ## chemical embedding: input Natoms x onehot (2개면 [1, 0] 아니면 [0, 1] 이런식)
    ## output : Natoms x feature dimension
    ## 2x0e -> 16x0e

    # add convnet layers
    # insertion preserves order
    for layer_i in range(num_layers):
        layers[f"layer{layer_i}_convnet"] = ConvNetLayer

    # .update also maintains insertion order
    layers.update(
        {
            # TODO: the next linear throws out all L > 0, don't create them in the last layer of convnet
            # -- output block --
            "conv_to_output_hidden": AtomwiseLinear,
            ## conv_to_output_hidden을 보면,
            ## input 은 n atoms x 272 << 272가 뭔진 모르겠음... layer갯수, l_max 같은 거랑 관련 있을 것 같음
            ## 16x0e+16x1e+16x2e+16x1o+16x2o << 이렇게 하면 16 * (1+3+9+1+3) 272임
            ## output은 n atoms x 8 << 왜 또 8인지 모름.)
            ## 
            
            "output_hidden_to_scalar": (
                AtomwiseLinear,
                dict(irreps_out="1x0e", out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY),
            ),
            ## n atomx x 8을 받아서 n atoms를 내놓음
            ## 8x0e -> 1x0e
            
        }
    )
    # Pooling per-atom energies of each atom
    layers["total_energy_sum"] = (
        AtomwiseReduce,
        dict(
            reduce="sum",
            field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
            out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
        ),
    )


    
    """
    def from_parameters(
        cls,
        shared_params: Mapping,
        layers: Dict[str, Union[Callable, Tuple[Callable, Dict[str, Any]]]],
        irreps_in: Optional[dict] = None,
    ):
    여기서 
    layers = {
        # -- Encode --
        "one_hot": OneHotAtomEncoding,
        "spharm_edges": SphericalHarmonicEdgeAttrs,
        "radial_basis": RadialBasisEdgeEncoding,
        # -- Embed features --
        "chemical_embedding": AtomwiseLinear,
        "layer{layer_i}_convnet" = ConvNetLayer
        "conv_to_output_hidden": AtomwiseLinear,
        "output_hidden_to_scalar": (
                AtomwiseLinear,
                dict(irreps_out="1x0e", out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY),
            )
        "total_energy_sum" : AtomwiseReduce
    }
    """
    
    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )
