import logging
from typing import Optional, List

import torch
import torch.nn.functional
from torch_runstats.scatter import scatter
import copy

from e3nn.o3 import Linear

from nequip.data import AtomicDataDict
from nequip.data.transforms import TypeMapper
from ._graph_mixin import GraphModuleMixin


class AtomwiseOperation(GraphModuleMixin, torch.nn.Module):
    def __init__(self, operation, field: str, irreps_in=None):
        super().__init__()
        self.operation = operation
        self.field = field
        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={field: operation.irreps_in},
            irreps_out={field: operation.irreps_out},
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data[self.field] = self.operation(data[self.field])
        return data


class AtomwiseLinear(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        field: str = AtomicDataDict.NODE_FEATURES_KEY,
        out_field: Optional[str] = None,
        irreps_in=None,
        irreps_out=None,
    ):
        super().__init__()
        self.field = field
        out_field = out_field if out_field is not None else field
        self.out_field = out_field
        if irreps_out is None:
            irreps_out = irreps_in[field]

        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[field],
            irreps_out={out_field: irreps_out},
        )
        self.linear = Linear(
            irreps_in=self.irreps_in[field], irreps_out=self.irreps_out[out_field]
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data[self.out_field] = self.linear(data[self.field])
        return data


class AtomwiseLinear_TCSM(GraphModuleMixin, torch.nn.Module):
    ## for conv_to_hidden



    def __init__(
        self,
        field: str = AtomicDataDict.NODE_FEATURES_KEY,
        out_field: Optional[str] = None,
        irreps_in=None,
        irreps_out=None,
    ):
        super().__init__()
     
        self.N=4
        #print("N")
        #print(self.N)

        self.field = field
        out_field = out_field if out_field is not None else field
        self.out_field = out_field
        if irreps_out is None:
            irreps_out = irreps_in[field]

        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[field],
            irreps_out={out_field: irreps_out},
        )

        self.linears = []
        for ii in range(self.N):
            self.linears.append(Linear(
                    irreps_in=self.irreps_in[field], irreps_out=self.irreps_out[out_field]
        ).cuda())
        
    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:

        if not "criterion_matrix" in data.keys():
            ################################################################
            ################################################################
            ## atomic envrionment descision tree로 설정
            edge_index_TCSM = data["edge_index"]
            atom_types_TCSM= data["atom_types"]
        
            count_TCSM = torch.bincount(edge_index_TCSM.reshape(-1)) # check how many edges the atom has.
                                                                      # for 5nm, the 10 would be good
            criterion_count = count_TCSM>10
            ## ratio 구분
            mixture = dict() # 0: pure N, 1: pure Si, 2: mixture
            for atom in range(len(atom_types_TCSM)):
                mixture[atom] = set()
            pairs = torch.transpose(edge_index_TCSM, 0, 1)
            pairs = pairs.detach().cpu().numpy()
            for pair in pairs:
                if atom_types_TCSM[pair[0]] != atom_types_TCSM[pair[1]]:
                    mixture[pair[0]].add(0)
                    mixture[pair[1]].add(0)
                else:
                    mixture[pair[0]].add(atom_types_TCSM[pair[0]].item()+1)
                    mixture[pair[1]].add(atom_types_TCSM[pair[1]].item()+1)
            criterion_mix = torch.zeros_like(criterion_count, dtype=int)
            #print(criterion_count.shape)
            #print(criterion_mix.shape)
            #print(mixture)
            window=num_feature//6
            for atom in range(len(atom_types_TCSM)):
                if len(mixture[atom])==0:
                    criterion_mix[atom] = 0
                else:    
                    criterion_mix[atom] += min(mixture[atom])
            #print("###"*10)
            #print("criterion_count")  
            #print(criterion_count)
            #print("###"*10)
            #print("criterion_mix")  
            #print(criterion_mix)
            #print(criterion_mix.shape)
            criterion= criterion_count *3 + criterion_mix
    
            
            #####################################################################
            ############ num_feature should be carefully selected. ##############
            #####################################################################
    
            
            num_feature=30
            window=num_feature//6
            criterion_matrix = torch.zeros((len(atom_types_TCSM), num_feature)).cuda()
            for ii in range(len(criterion)):
                start_idx = criterion[ii] * window
                end_idx = start_idx + window
                criterion_matrix[ii, start_idx:end_idx] = 1
            #print("###"*10)
            #print("criterion_matrix")        
            #print(criterion_matrix)
            #print(criterion_matrix.shape)
            data["criterion_matrix"]=criterion_matrix
            
        

        data[self.out_field] = torch.mul(self.linears[0](data[self.field]), data["criterion_matrix"])
        #print("###"*10)
        #print("data")
        #print(data[self.out_field])
        #print(data[self.out_field].shape)
        return data


class AtomwiseLinear_Nlinears(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        field: str = AtomicDataDict.NODE_FEATURES_KEY,
        out_field: Optional[str] = None,
        irreps_in=None,
        irreps_out=None,
        # num_atom_types: int = 3,  # Initialize with the number of atom types
    ):
        super().__init__()
        self.N = 3  # Initialize self.N with the number of atom types
        self.field = field
        out_field = out_field if out_field is not None else field
        self.out_field = out_field
        if irreps_out is None:
            irreps_out = irreps_in[field]

        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[field],
            irreps_out={out_field: irreps_out},
        )

        # Create a list of linear layers for each atom type
        self.linear1 = Linear(irreps_in=self.irreps_in[field], irreps_out=self.irreps_out[out_field]).cuda()
        self.linear2 = Linear(irreps_in=self.irreps_in[field], irreps_out=self.irreps_out[out_field]).cuda()
        self.linear3 = Linear(irreps_in=self.irreps_in[field], irreps_out=self.irreps_out[out_field]).cuda()
        # self.linears = [
        #     Linear(irreps_in=self.irreps_in[field], irreps_out=self.irreps_out[out_field]).cuda()
        #     for _ in range(self.N)
        # ]

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:


        num_atoms, num_features = data[self.field].shape

        outfeature=32
 
        # threshold=25
        # if data["now_epochs"] > threshold:

        out = torch.zeros((self.N, num_atoms, outfeature), device=data[self.field].device)
        
        out[0] = self.linear1(data[self.field])
        out[1] = self.linear2(data[self.field])
        out[2] = self.linear3(data[self.field])
        # out = torch.stack([linear(data[self.field]) for linear in self.linears].cuda(), dim=0)
        data[self.out_field] = torch.sum(torch.mul(out,torch.unsqueeze(data["one_hot_criterion_matrix"].T, dim=2)), dim=0)#.to(device=data[self.field].device)
        return data

class AtomwiseLinear_Nlinears_pretrain(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        field: str = AtomicDataDict.NODE_FEATURES_KEY,
        out_field: Optional[str] = None,
        irreps_in=None,
        irreps_out=None,
        # num_atom_types: int = 3,  # Initialize with the number of atom types
    ):
        super().__init__()
        self.N = 3  # Initialize self.N with the number of atom types
        self.field = field
        out_field = out_field if out_field is not None else field
        self.out_field = out_field
        if irreps_out is None:
            irreps_out = irreps_in[field]

        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[field],
            irreps_out={out_field: irreps_out},
        )

        # Create a list of linear layers for each atom type
        self.linear1 = Linear(irreps_in=self.irreps_in[field], irreps_out=self.irreps_out[out_field]).cuda()
        self.linear2 = Linear(irreps_in=self.irreps_in[field], irreps_out=self.irreps_out[out_field]).cuda()
        self.linear3 = Linear(irreps_in=self.irreps_in[field], irreps_out=self.irreps_out[out_field]).cuda()
        # self.linears = [
        #     Linear(irreps_in=self.irreps_in[field], irreps_out=self.irreps_out[out_field]).cuda()
        #     for _ in range(self.N)
        # ]

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:


        num_atoms, num_features = data[self.field].shape

        outfeature=32
 
        # threshold=25
        # if data["now_epochs"] > threshold:

        out = torch.zeros((self.N, num_atoms, outfeature), device=data[self.field].device)
        
        out[0] = self.linear1(data[self.field])
        out[1] = self.linear2(data[self.field])
        out[2] = self.linear3(data[self.field])
        # out = torch.stack([linear(data[self.field]) for linear in self.linears].cuda(), dim=0)
        data[self.out_field] = out[0]#.to(device=data[self.field].device)
        return data

class AtomwiseReduce(GraphModuleMixin, torch.nn.Module):
    constant: float

    def __init__(
        self,
        field: str,
        out_field: Optional[str] = None,
        reduce="sum",
        avg_num_atoms=None,
        irreps_in={},
    ):
        super().__init__()
        assert reduce in ("sum", "mean", "normalized_sum")
        self.constant = 1.0
        if reduce == "normalized_sum":
            assert avg_num_atoms is not None
            self.constant = float(avg_num_atoms) ** -0.5
            reduce = "sum"
        self.reduce = reduce
        self.field = field
        self.out_field = f"{reduce}_{field}" if out_field is None else out_field
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: irreps_in[self.field]}
            if self.field in irreps_in
            else {},
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = AtomicDataDict.with_batch(data)
        data[self.out_field] = scatter(
            data[self.field], data[AtomicDataDict.BATCH_KEY], dim=0, reduce=self.reduce
        )
        if self.constant != 1.0:
            data[self.out_field] = data[self.out_field] * self.constant
        return data


class PerSpeciesScaleShift(GraphModuleMixin, torch.nn.Module):
    """Scale and/or shift a predicted per-atom property based on (learnable) per-species/type parameters.

    Args:
        field: the per-atom field to scale/shift.
        num_types: the number of types in the model.
        shifts: the initial shifts to use, one per atom type.
        scales: the initial scales to use, one per atom type.
        arguments_in_dataset_units: if ``True``, says that the provided shifts/scales are in dataset
            units (in which case they will be rescaled appropriately by any global rescaling later
            applied to the model); if ``False``, the provided shifts/scales will be used without modification.

            For example, if identity shifts/scales of zeros and ones are provided, this should be ``False``.
            But if scales/shifts computed from the training data are used, and are thus in dataset units,
            this should be ``True``.
        out_field: the output field; defaults to ``field``.
    """

    field: str
    out_field: str
    scales_trainble: bool
    shifts_trainable: bool
    has_scales: bool
    has_shifts: bool

    def __init__(
        self,
        field: str,
        num_types: int,
        type_names: List[str],
        shifts: Optional[List[float]],
        scales: Optional[List[float]],
        arguments_in_dataset_units: bool,
        out_field: Optional[str] = None,
        scales_trainable: bool = False,
        shifts_trainable: bool = False,
        irreps_in={},
    ):
        super().__init__()
        self.num_types = num_types
        self.type_names = type_names
        self.field = field
        self.out_field = f"shifted_{field}" if out_field is None else out_field
        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={self.field: "0e"},  # input to shift must be a single scalar
            irreps_out={self.out_field: irreps_in[self.field]},
        )

        self.has_shifts = shifts is not None
        if shifts is not None:
            shifts = torch.as_tensor(shifts, dtype=torch.get_default_dtype())
            if len(shifts.reshape([-1])) == 1:
                shifts = torch.ones(num_types) * shifts
            assert shifts.shape == (num_types,), f"Invalid shape of shifts {shifts}"
            self.shifts_trainable = shifts_trainable
            if shifts_trainable:
                self.shifts = torch.nn.Parameter(shifts)
            else:
                self.register_buffer("shifts", shifts)

        self.has_scales = scales is not None
        if scales is not None:
            scales = torch.as_tensor(scales, dtype=torch.get_default_dtype())
            if len(scales.reshape([-1])) == 1:
                scales = torch.ones(num_types) * scales
            assert scales.shape == (num_types,), f"Invalid shape of scales {scales}"
            self.scales_trainable = scales_trainable
            if scales_trainable:
                self.scales = torch.nn.Parameter(scales)
            else:
                self.register_buffer("scales", scales)

        self.arguments_in_dataset_units = arguments_in_dataset_units

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:

        if not (self.has_scales or self.has_shifts):
            return data

        species_idx = data[AtomicDataDict.ATOM_TYPE_KEY]
        in_field = data[self.field]
        assert len(in_field) == len(
            species_idx
        ), "in_field doesnt seem to have correct per-atom shape"
        if self.has_scales:
            in_field = self.scales[species_idx].view(-1, 1) * in_field
        if self.has_shifts:
            in_field = self.shifts[species_idx].view(-1, 1) + in_field
        data[self.out_field] = in_field
        return data

    def update_for_rescale(self, rescale_module):
        if hasattr(rescale_module, "related_scale_keys"):
            if self.out_field not in rescale_module.related_scale_keys:
                return
        if self.arguments_in_dataset_units and rescale_module.has_scale:
            logging.debug(
                f"PerSpeciesScaleShift's arguments were in dataset units; rescaling:\n  "
                f"Original scales: {TypeMapper.format(self.scales, self.type_names) if self.has_scales else 'n/a'} "
                f"shifts: {TypeMapper.format(self.shifts, self.type_names) if self.has_shifts else 'n/a'}"
            )
            with torch.no_grad():
                if self.has_scales:
                    self.scales.div_(rescale_module.scale_by)
                if self.has_shifts:
                    self.shifts.div_(rescale_module.scale_by)
            logging.debug(
                f"  New scales: {TypeMapper.format(self.scales, self.type_names) if self.has_scales else 'n/a'} "
                f"shifts: {TypeMapper.format(self.shifts, self.type_names) if self.has_shifts else 'n/a'}"
            )
