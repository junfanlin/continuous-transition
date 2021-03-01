import torch
import numpy as np
from numbers import Number
from typing import Union, Optional

from tianshou.data.batch import _parse_value, Batch


def to_numpy(x: Union[
    Batch, dict, list, tuple, np.ndarray, torch.Tensor]) -> Union[
        Batch, dict, list, tuple, np.ndarray, torch.Tensor]:
    """Return an object without torch.Tensor."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    elif isinstance(x, dict):
        for k, v in x.items():
            x[k] = to_numpy(v)
    elif isinstance(x, Batch):
        x.to_numpy()
    elif isinstance(x, (list, tuple)):
        try:
            x = to_numpy(_parse_value(x))
        except TypeError:
            x = [to_numpy(e) for e in x]
    else:  # fallback
        x = np.asanyarray(x)
    return x


def to_torch(x: Union[Batch, dict, list, tuple, np.ndarray, torch.Tensor],
             dtype: Optional[torch.dtype] = None,
             device: Union[str, int, torch.device] = 'cpu'
             ) -> Union[Batch, dict, list, tuple, np.ndarray, torch.Tensor]:
    """Return an object without np.ndarray."""
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        x = x.to(device)
    elif isinstance(x, dict):
        for k, v in x.items():
            x[k] = to_torch(v, dtype, device)
    elif isinstance(x, Batch):
        x.to_torch(dtype, device)
    elif isinstance(x, (np.number, np.bool_, Number)):
        x = to_torch(np.asanyarray(x), dtype, device)
    elif isinstance(x, (list, tuple)):
        try:
            x = to_torch(_parse_value(x), dtype, device)
        except TypeError:
            x = [to_torch(e, dtype, device) for e in x]
    else:  # fallback
        x = np.asanyarray(x)
        if issubclass(x.dtype.type, (np.bool_, np.number)):
            x = torch.from_numpy(x).to(device)
            if dtype is not None:
                x = x.type(dtype)
        else:
            raise TypeError(f"object {x} cannot be converted to torch.")
    return x


def to_torch_as(x: Union[torch.Tensor, dict, Batch, np.ndarray],
                y: torch.Tensor
                ) -> Union[dict, Batch, torch.Tensor]:
    """Return an object without np.ndarray. Same as
    ``to_torch(x, dtype=y.dtype, device=y.device)``.
    """
    assert isinstance(y, torch.Tensor)
    return to_torch(x, dtype=y.dtype, device=y.device)

# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt

def simulate_beta(alpha, size=1, rng=np.random):
    """
    from binary to uniform
    alpha (0, 1]
    """
    # return (rng.randn(size) * alpha / 2) % 1
    return (torch.sigmoid(torch.randn(size) * alpha * 2) - 0.5) % 1


# data = simulate_beta(0.9, 10000)
#
#
# data = torch.distributions.beta.Beta(5e-1, 1).rsample([10000,])
#
# plt.hist(data.numpy())
# plt.show()


