from torch.nn import Module, Linear
from torch.utils import swap_tensors
from typing import Protocol

class LinearConstructor(Protocol):
    @staticmethod
    def __call__(
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> Linear: ...

def replace_linear(module: Module, linear_ctor: LinearConstructor) -> None:
    """
    Replace all nn.Linear modules with a custom subclass of Linear
    Usage:
    # pip install -e 'git+https://github.com/sekstini/gpupoor.git#egg=gpu_poor' --config-settings editable_mode=compat
    from gpu_poor.modules.lowp_linear import LowPrecisionLinear
    replace_linear(f16_enc, LowPrecisionLinear)
    """
    for child_name, child_mod in module.named_children():
        if isinstance(child_mod, Linear):
            lp_lin = linear_ctor(
                in_features=child_mod.in_features,
                out_features=child_mod.out_features,
                bias=child_mod.bias is not None,
                device='meta',
                dtype=child_mod.weight.dtype,
            )
            swap_tensors(lp_lin.weight, child_mod.weight)
            if child_mod.bias is not None:
                swap_tensors(lp_lin.bias, child_mod.bias)
            setattr(module, child_name, lp_lin)
        else:
            replace_linear(child_mod, linear_ctor)