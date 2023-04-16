import pickle
from abc import ABC, abstractmethod
from os import PathLike
from typing import Dict, List, Type

import torch

from nnsmith.abstract.op import AbsOpBase, AbsTensor
from nnsmith.gir import GraphIR
from nnsmith.materialize import Model, Oracle
from nnsmith.materialize.torch.forward import ALL_TORCH_OPS
from nnsmith.materialize.torch.input_gen import PracticalHybridSearch
from nnsmith.materialize.torch.symbolnet import SymbolNet
from nnsmith.util import register_seed_setter


class TorchModel(Model, ABC):
    def __init__(self) -> None:
        super().__init__()
        self.torch_model: SymbolNet = None
        self.sat_inputs = None

    @classmethod
    @abstractmethod
    def device(cls) -> torch.device:
        pass

    @property
    def version(self) -> str:
        return torch.__version__

    @classmethod
    def from_gir(cls: Type["TorchModel"], ir: GraphIR, **kwargs) -> "TorchModel":
        ret = cls()
        ret.torch_model = SymbolNet(ir, **kwargs).to(cls.device())
        return ret

    @staticmethod
    def gir_name() -> str:
        return "gir.pkl"

    def refine_weights(self) -> None:
        self.torch_model.enable_proxy_grad()
        use_cuda = self.device().type == "cuda"
        searcher = PracticalHybridSearch(self.torch_model, use_cuda=use_cuda)
        # TODO(@ganler): Can we directly get both inputs and outputs?
        _, inputs = searcher.search(
            max_time_ms=20,
            max_sample=2,
        )
        if inputs:
            self.sat_inputs = inputs
        self.torch_model.disable_proxy_grad()

    def make_oracle(self) -> Oracle:
        with torch.no_grad():
            self.torch_model.eval()
            # fall back to random inputs if no solution is found.
            if self.sat_inputs is None:
                inputs = self.torch_model.get_random_inps()
            else:
                inputs = self.sat_inputs
            outputs = self.torch_model.forward(
                *[v.to(self.device()) for _, v in inputs.items()]
            )

        # numpyify
        input_dict = {k: v.cpu().detach().numpy() for k, v in inputs.items()}
        output_dict = {}
        for oname, val in zip(self.output_like.keys(), outputs):
            output_dict[oname] = val.cpu().detach().numpy()

        # add grad to output_dict
        if requires_grad:
            for name, param in grad_var_list:
                if param.data.is_floating_point():
                    if param.grad == None:
                        output_dict['grad_' + name] = None
                    else:
                        output_dict['grad_' + name] = param.grad.cpu().detach().numpy()
                    DTEST_LOG.info(f"get grad from {name}")
                    param.requires_grad = False
            return Oracle(input_dict, output_dict, provider="torch[cpu] eager-ad")
        else:
            return Oracle(input_dict, output_dict, provider="torch[cpu] eager")

    def dump(self, path: PathLike):
        torch.save(self.torch_model.state_dict(), path)
        gir_path = path.replace(
            TorchModel.name_prefix() + TorchModel.name_suffix(),
            TorchModel.gir_name(),
        )
        with open(gir_path, "wb") as f:
            pickle.dump(self.torch_model.ir, f)

    @classmethod
    def load(cls, path: PathLike) -> "TorchModel":
        ret = cls()
        gir_path = path.replace(
            cls.name_prefix() + cls.name_suffix(),
            cls.gir_name(),
        )
        with open(gir_path, "rb") as f:
            ir = pickle.load(f)
        torch_model = SymbolNet(ir).to(cls.device())
        torch_model.load_state_dict(torch.load(path), strict=False)
        ret.torch_model = torch_model
        return ret

    @staticmethod
    def name_suffix() -> str:
        return ".pth"

    @property
    def input_like(self) -> Dict[str, AbsTensor]:
        return self.torch_model.input_like

    @property
    def output_like(self) -> Dict[str, AbsTensor]:
        return self.torch_model.output_like

    @property
    def native_model(self) -> SymbolNet:
        return self.torch_model

    @staticmethod
    def operators() -> List[Type[AbsOpBase]]:
        return ALL_TORCH_OPS

    @staticmethod
    def add_seed_setter() -> None:
        register_seed_setter("torch", torch.manual_seed, overwrite=True)


class TorchModelCPU(TorchModel):
    @classmethod
    def device(cls) -> torch.device:
        return torch.device("cpu")


class TorchModelCUDA(TorchModel):
    @classmethod
    def device(cls) -> torch.device:
        return torch.device("cuda")
