# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import functools
import inspect
from abc import ABC, abstractmethod
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple, Union
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union
from typing import Any, Dict, Hashable, Iterable, List, Optional, Sequence, Tuple, Union
# import orderred dict
from collections import OrderedDict
from torch.nn import Module, ModuleDict
from functools import partial, wraps
import warnings

import torch
from torch import Tensor
from torch.nn import Module
from typing_extensions import Literal



_TORCH_GREATER_EQUAL_1_12: True
_XLA_AVAILABLE: True
def _fbeta_reduce(
    tp: Tensor,
    fp: Tensor,
    tn: Tensor,
    fn: Tensor,
    beta: float,
    average: Optional[Literal["binary", "micro", "macro", "weighted", "none"]],
    multidim_average: Literal["global", "samplewise"] = "global",
) -> Tensor:
    beta2 = beta**2
    if average == "binary":
        return _safe_divide((1 + beta2) * tp, (1 + beta2) * tp + beta2 * fn + fp)
    elif average == "micro":
        tp = tp.sum(dim=0 if multidim_average == "global" else 1)
        fn = fn.sum(dim=0 if multidim_average == "global" else 1)
        fp = fp.sum(dim=0 if multidim_average == "global" else 1)
        return _safe_divide((1 + beta2) * tp, (1 + beta2) * tp + beta2 * fn + fp)
    else:
        fbeta_score = _safe_divide((1 + beta2) * tp, (1 + beta2) * tp + beta2 * fn + fp)
        if average is None or average == "none":
            return fbeta_score
        if average == "weighted":
            weights = tp + fn
        else:
            weights = torch.ones_like(fbeta_score)
        return _safe_divide(weights * fbeta_score, weights.sum(-1, keepdim=True)).sum(-1)
def _simple_gather_all_tensors(result: Tensor, group: Any, world_size: int) -> List[Tensor]:
    gathered_result = [torch.zeros_like(result) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_result, result, group)
    return gathered_result

def gather_all_tensors(result: Tensor, group: Optional[Any] = None) -> List[Tensor]:
    """Function to gather all tensors from several ddp processes onto a list that is broadcasted to all processes.
    Works on tensors that have the same number of dimensions, but where each dimension may differ. In this case
    tensors are padded, gathered and then trimmed to secure equal workload for all processes.

    Args:
        result: the value to sync
        group: the process group to gather results from. Defaults to all processes (world)

    Return:
        gathered_result: list with size equal to the process group where
            ``gathered_result[i]`` corresponds to result tensor from process ``i``
    """
    if group is None:
        group = torch.distributed.group.WORLD

    # convert tensors to contiguous format
    result = result.contiguous()

    world_size = torch.distributed.get_world_size(group)
    torch.distributed.barrier(group=group)

    # if the tensor is scalar, things are easy
    if result.ndim == 0:
        return _simple_gather_all_tensors(result, group, world_size)

    # 1. Gather sizes of all tensors
    local_size = torch.tensor(result.shape, device=result.device)
    local_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    torch.distributed.all_gather(local_sizes, local_size, group=group)
    max_size = torch.stack(local_sizes).max(dim=0).values
    all_sizes_equal = all(all(ls == max_size) for ls in local_sizes)

    # 2. If shapes are all the same, then do a simple gather:
    if all_sizes_equal:
        return _simple_gather_all_tensors(result, group, world_size)

    # 3. If not, we need to pad each local tensor to maximum size, gather and then truncate
    pad_dims = []
    pad_by = (max_size - local_size).detach().cpu()
    for val in reversed(pad_by):
        pad_dims.append(0)
        pad_dims.append(val.item())
    result_padded = F.pad(result, pad_dims)
    gathered_result = [torch.zeros_like(result_padded) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_result, result_padded, group)
    for idx, item_size in enumerate(local_sizes):
        slice_param = [slice(dim_size) for dim_size in item_size]
        gathered_result[idx] = gathered_result[idx][slice_param]
    return gathered_result


def select_topk(prob_tensor: Tensor, topk: int = 1, dim: int = 1) -> Tensor:
    """Convert a probability tensor to binary by selecting top-k the highest entries.

    Args:
        prob_tensor: dense tensor of shape ``[..., C, ...]``, where ``C`` is in the
            position defined by the ``dim`` argument
        topk: number of the highest entries to turn into 1s
        dim: dimension on which to compare entries

    Returns:
        A binary tensor of the same shape as the input tensor of type ``torch.int32``

    Example:
        >>> x = torch.tensor([[1.1, 2.0, 3.0], [2.0, 1.0, 0.5]])
        >>> select_topk(x, topk=2)
        tensor([[0, 1, 1],
                [1, 1, 0]], dtype=torch.int32)
    """
    zeros = torch.zeros_like(prob_tensor)
    if topk == 1:  # argmax has better performance than topk
        topk_tensor = zeros.scatter(dim, prob_tensor.argmax(dim=dim, keepdim=True), 1.0)
    else:
        topk_tensor = zeros.scatter(dim, prob_tensor.topk(k=topk, dim=dim).indices, 1.0)
    return topk_tensor.int()


def _squeeze_scalar_element_tensor(x: Tensor) -> Tensor:
    return x.squeeze() if x.numel() == 1 else x

def _squeeze_if_scalar(data: Any) -> Any:
    return apply_to_collection(data, Tensor, _squeeze_scalar_element_tensor)


def _bincount(x: Tensor, minlength: Optional[int] = None) -> Tensor:
    """PyTorch currently does not support``torch.bincount`` for:

        - deterministic mode on GPU.
        - MPS devices

    This implementation fallback to a for-loop counting occurrences in that case.

    Args:
        x: tensor to count
        minlength: minimum length to count

    Returns:
        Number of occurrences for each unique element in x
    """
    _XLA_AVAILABLE= True

    if minlength is None:
        minlength = len(torch.unique(x))
    if torch.are_deterministic_algorithms_enabled() or _XLA_AVAILABLE or _TORCH_GREATER_EQUAL_1_12 and x.is_mps:
        output = torch.zeros(minlength, device=x.device, dtype=torch.long)
        for i in range(minlength):
            output[i] = (x == i).sum()
        return output
    return torch.bincount(x, minlength=minlength)


def _multiclass_stat_scores_tensor_validation(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
) -> None:
    """Validate tensor input.

    - if target has one more dimension than preds, then all dimensions except for preds.shape[1] should match
    exactly. preds.shape[1] should have size equal to number of classes
    - if preds and target have same number of dims, then all dimensions should match
    - if ``multidim_average`` is set to ``samplewise`` preds tensor needs to be atleast 2 dimensional in the
    int case and 3 dimensional in the float case
    - all values in target tensor that are not ignored have to be {0, ..., num_classes - 1}
    - if pred tensor is not floating point, then all values also have to be in {0, ..., num_classes - 1}
    """
    if preds.ndim == target.ndim + 1:
        if not preds.is_floating_point():
            raise ValueError("If `preds` have one dimension more than `target`, `preds` should be a float tensor.")
        if preds.shape[1] != num_classes:
            raise ValueError(
                "If `preds` have one dimension more than `target`, `preds.shape[1]` should be"
                " equal to number of classes."
            )
        if preds.shape[2:] != target.shape[1:]:
            raise ValueError(
                "If `preds` have one dimension more than `target`, the shape of `preds` should be"
                " (N, C, ...), and the shape of `target` should be (N, ...)."
            )
        if multidim_average != "global" and preds.ndim < 3:
            raise ValueError(
                "If `preds` have one dimension more than `target`, the shape of `preds` should "
                " atleast 3D when multidim_average is set to `samplewise`"
            )

    elif preds.ndim == target.ndim:
        if preds.shape != target.shape:
            raise ValueError(
                "The `preds` and `target` should have the same shape,",
                f" got `preds` with shape={preds.shape} and `target` with shape={target.shape}.",
            )
        if multidim_average != "global" and preds.ndim < 2:
            raise ValueError(
                "When `preds` and `target` have the same shape, the shape of `preds` should "
                " atleast 2D when multidim_average is set to `samplewise`"
            )
    else:
        raise ValueError(
            "Either `preds` and `target` both should have the (same) shape (N, ...), or `target` should be (N, ...)"
            " and `preds` should be (N, C, ...)."
        )

    num_unique_values = len(torch.unique(target))
    if ignore_index is None:
        check = num_unique_values > num_classes
    else:
        check = num_unique_values > num_classes + 1
    if check:
        raise RuntimeError(
            "Detected more unique values in `target` than `num_classes`. Expected only "
            f"{num_classes if ignore_index is None else num_classes + 1} but found"
            f"{num_unique_values} in `target`."
        )

    if not preds.is_floating_point():
        unique_values = torch.unique(preds)
        if len(unique_values) > num_classes:
            raise RuntimeError(
                "Detected more unique values in `preds` than `num_classes`. Expected only "
                f"{num_classes} but found {len(unique_values)} in `preds`."
            )


def _multiclass_stat_scores_format(
    preds: Tensor,
    target: Tensor,
    top_k: int = 1,
) -> Tuple[Tensor, Tensor]:
    """Convert all input to label format except if ``top_k`` is not 1.

    - Applies argmax if preds have one more dimension than target
    - Flattens additional dimensions
    """
    # Apply argmax if we have one more dimension
    if preds.ndim == target.ndim + 1 and top_k == 1:
        preds = preds.argmax(dim=1)
    if top_k != 1:
        preds = preds.reshape(*preds.shape[:2], -1)
    else:
        preds = preds.reshape(preds.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    return preds, target


def _multiclass_stat_scores_update(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    top_k: int = 1,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Computes the statistics.

    - If ``multidim_average`` is equal to samplewise or ``top_k`` is not 1, we transform both preds and
    target into one hot format.
    - Else we calculate statistics by first calculating the confusion matrix and afterwards deriving the
    statistics from that
    - Remove all datapoints that should be ignored. Depending on if ``ignore_index`` is in the set of labels
    or outside we have do use different augmentation stategies when one hot encoding.
    """
    if multidim_average == "samplewise" or top_k != 1:
        ignore_in = 0 <= ignore_index <= num_classes - 1 if ignore_index is not None else None
        if ignore_index is not None and not ignore_in:
            preds = preds.clone()
            target = target.clone()
            idx = target == ignore_index
            target[idx] = num_classes
            idx = idx.unsqueeze(1).repeat(1, num_classes, 1) if preds.ndim > target.ndim else idx
            preds[idx] = num_classes

        if top_k > 1:
            preds_oh = torch.movedim(select_topk(preds, topk=top_k, dim=1), 1, -1)
        else:
            preds_oh = torch.nn.functional.one_hot(
                preds, num_classes + 1 if ignore_index is not None and not ignore_in else num_classes
            )
        target_oh = torch.nn.functional.one_hot(
            target, num_classes + 1 if ignore_index is not None and not ignore_in else num_classes
        )
        if ignore_index is not None:
            if 0 <= ignore_index <= num_classes - 1:
                target_oh[target == ignore_index, :] = -1
            else:
                preds_oh = preds_oh[..., :-1] if top_k == 1 else preds_oh
                target_oh = target_oh[..., :-1]
                target_oh[target == num_classes, :] = -1
        sum_dim = [0, 1] if multidim_average == "global" else [1]
        tp = ((target_oh == preds_oh) & (target_oh == 1)).sum(sum_dim)
        fn = ((target_oh != preds_oh) & (target_oh == 1)).sum(sum_dim)
        fp = ((target_oh != preds_oh) & (target_oh == 0)).sum(sum_dim)
        tn = ((target_oh == preds_oh) & (target_oh == 0)).sum(sum_dim)
    elif average == "micro":
        preds = preds.flatten()
        target = target.flatten()
        if ignore_index is not None:
            idx = target != ignore_index
            preds = preds[idx]
            target = target[idx]
        tp = (preds == target).sum()
        fp = (preds != target).sum()
        fn = (preds != target).sum()
        tn = num_classes * preds.numel() - (fp + fn + tp)
    else:
        preds = preds.flatten()
        target = target.flatten()
        if ignore_index is not None:
            idx = target != ignore_index
            preds = preds[idx]
            target = target[idx]
        unique_mapping = target.to(torch.long) * num_classes + preds.to(torch.long)
        bins = _bincount(unique_mapping, minlength=num_classes**2)
        confmat = bins.reshape(num_classes, num_classes)
        tp = confmat.diag()
        fp = confmat.sum(0) - tp
        fn = confmat.sum(1) - tp
        tn = confmat.sum() - (fp + fn + tp)
    return tp, fp, tn, fn

def _multiclass_fbeta_score_arg_validation(
    beta: float,
    num_classes: int,
    top_k: int = 1,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
) -> None:
    if not (isinstance(beta, float) and beta > 0):
        raise ValueError(f"Expected argument `beta` to be a float larger than 0, but got {beta}.")
    _multiclass_stat_scores_arg_validation(num_classes, top_k, average, multidim_average, ignore_index)


def _multiclass_stat_scores_arg_validation(
    num_classes: int,
    top_k: int = 1,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
) -> None:
    """Validate non tensor input.

    - ``num_classes`` has to be a int larger than 1
    - ``top_k`` has to be an int larger than 0 but no larger than number of classes
    - ``average`` has to be "micro" | "macro" | "weighted" | "none"
    - ``multidim_average`` has to be either "global" or "samplewise"
    - ``ignore_index`` has to be None or int
    """
    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError(f"Expected argument `num_classes` to be an integer larger than 1, but got {num_classes}")
    if not isinstance(top_k, int) and top_k < 1:
        raise ValueError(f"Expected argument `top_k` to be an integer larger than or equal to 1, but got {top_k}")
    if top_k > num_classes:
        raise ValueError(
            f"Expected argument `top_k` to be smaller or equal to `num_classes` but got {top_k} and {num_classes}"
        )
    allowed_average = ("micro", "macro", "weighted", "none", None)
    if average not in allowed_average:
        raise ValueError(f"Expected argument `average` to be one of {allowed_average}, but got {average}")
    allowed_multidim_average = ("global", "samplewise")
    if multidim_average not in allowed_multidim_average:
        raise ValueError(
            f"Expected argument `multidim_average` to be one of {allowed_multidim_average}, but got {multidim_average}"
        )
    if ignore_index is not None and not isinstance(ignore_index, int):
        raise ValueError(f"Expected argument `ignore_index` to either be `None` or an integer, but got {ignore_index}")


def _multiclass_stat_scores_compute(
    tp: Tensor,
    fp: Tensor,
    tn: Tensor,
    fn: Tensor,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
    multidim_average: Literal["global", "samplewise"] = "global",
) -> Tensor:
    """Stack statistics and compute support also.

    Applies average strategy afterwards.
    """
    res = torch.stack([tp, fp, tn, fn, tp + fn], dim=-1)
    sum_dim = 0 if multidim_average == "global" else 1
    if average == "micro":
        return res.sum(sum_dim) if res.ndim > 1 else res
    if average == "macro":
        return res.float().mean(sum_dim)
    elif average == "weighted":
        weight = tp + fn
        if multidim_average == "global":
            return (res * (weight / weight.sum()).reshape(*weight.shape, 1)).sum(sum_dim)
        else:
            return (res * (weight / weight.sum(-1, keepdim=True)).reshape(*weight.shape, 1)).sum(sum_dim)
    elif average is None or average == "none":
        return res
    
def _safe_divide(num: Tensor, denom: Tensor) -> Tensor:
    """Safe division, by preventing division by zero.

    Additionally casts to float if input is not already to secure backwards compatibility.
    """
    denom[denom == 0.0] = 1
    num = num if num.is_floating_point() else num.float()
    denom = denom if denom.is_floating_point() else denom.float()
    return num / denom

def _warn(*args: Any, **kwargs: Any) -> None:
    warnings.warn(*args, **kwargs)


def allclose(tensor1: Tensor, tensor2: Tensor) -> bool:
    """Wrapper of torch.allclose that is robust towards dtype difference."""
    if tensor1.dtype != tensor2.dtype:
        tensor2 = tensor2.to(dtype=tensor1.dtype)
    return torch.allclose(tensor1, tensor2)


class TorchMetricsUserError(Exception):
    """Error used to inform users of a wrong combination of Metric API calls."""

def dim_zero_cat(x: Union[Tensor, List[Tensor]]) -> Tensor:
    """Concatenation along the zero dimension."""
    x = x if isinstance(x, (list, tuple)) else [x]
    x = [y.unsqueeze(0) if y.numel() == 1 and y.ndim == 0 else y for y in x]
    if not x:  # empty list
        raise ValueError("No samples to concatenate")
    return torch.cat(x, dim=0)

def dim_zero_sum(x: Tensor) -> Tensor:
    """Summation along the zero dimension."""
    return torch.sum(x, dim=0)


def dim_zero_mean(x: Tensor) -> Tensor:
    """Average along the zero dimension."""
    return torch.mean(x, dim=0)


def dim_zero_max(x: Tensor) -> Tensor:
    """Max along the zero dimension."""
    return torch.max(x, dim=0).values


def dim_zero_min(x: Tensor) -> Tensor:
    """Min along the zero dimension."""
    return torch.min(x, dim=0).values


def _flatten(x: Sequence) -> list:
    """Flatten list of list into single list."""
    return [item for sublist in x for item in sublist]

def _flatten_dict(x: Dict) -> Dict:
    """Flatten dict of dicts into single dict."""
    new_dict = {}
    for key, value in x.items():
        if isinstance(value, dict):
            for k, v in value.items():
                new_dict[k] = v
        else:
            new_dict[key] = value
    return new_dict

def apply_to_collection(
    data: Any,
    dtype: Union[type, tuple],
    function: Callable,
    *args: Any,
    wrong_dtype: Optional[Union[type, tuple]] = None,
    **kwargs: Any,
) -> Any:
    """Recursively applies a function to all elements of a certain dtype.

    Args:
        data: the collection to apply the function to
        dtype: the given function will be applied to all elements of this dtype
        function: the function to apply
        *args: positional arguments (will be forwarded to call of ``function``)
        wrong_dtype: the given function won't be applied if this type is specified and the given collections is of
            the :attr:`wrong_type` even if it is of type :attr`dtype`
        **kwargs: keyword arguments (will be forwarded to call of ``function``)

    Returns:
        the resulting collection

    Example:
        >>> apply_to_collection(torch.tensor([8, 0, 2, 6, 7]), dtype=Tensor, function=lambda x: x ** 2)
        tensor([64,  0,  4, 36, 49])
        >>> apply_to_collection([8, 0, 2, 6, 7], dtype=int, function=lambda x: x ** 2)
        [64, 0, 4, 36, 49]
        >>> apply_to_collection(dict(abc=123), dtype=int, function=lambda x: x ** 2)
        {'abc': 15129}
    """
    elem_type = type(data)

    # Breaking condition
    if isinstance(data, dtype) and (wrong_dtype is None or not isinstance(data, wrong_dtype)):
        return function(data, *args, **kwargs)

    # Recursively apply to collection items
    if isinstance(data, Mapping):
        return elem_type({k: apply_to_collection(v, dtype, function, *args, **kwargs) for k, v in data.items()})

    if isinstance(data, tuple) and hasattr(data, "_fields"):  # named tuple
        return elem_type(*(apply_to_collection(d, dtype, function, *args, **kwargs) for d in data))

    if isinstance(data, Sequence) and not isinstance(data, str):
        return elem_type([apply_to_collection(d, dtype, function, *args, **kwargs) for d in data])

    # data is neither of dtype, nor a collection
    return data

# from torchmetrics.utilities.distributed import gather_all_tensors
# from torchmetrics.utilities.exceptions import TorchMetricsUserError


def jit_distributed_available() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


class Metric(Module, ABC):
    """Base class for all metrics present in the Metrics API.

    Implements ``add_state()``, ``forward()``, ``reset()`` and a few other things to
    handle distributed synchronization and per-step metric computation.

    Override ``update()`` and ``compute()`` functions to implement your own metric. Use
    ``add_state()`` to register metric state variables which keep track of state on each
    call of ``update()`` and are synchronized across processes when ``compute()`` is called.

    Note:
        Metric state variables can either be :class:`~torch.Tensor` or an empty list which can we used
        to store :class:`~torch.Tensor`.

    Note:
        Different metrics only override ``update()`` and not ``forward()``. A call to ``update()``
        is valid, but it won't return the metric value at the current step. A call to ``forward()``
        automatically calls ``update()`` and also returns the metric value at the current step.

    Args:
        kwargs: additional keyword arguments, see :ref:`Metric kwargs` for more info.

            - compute_on_cpu: If metric state should be stored on CPU during computations. Only works
                for list states.
            - dist_sync_on_step: If metric state should synchronize on ``forward()``. Default is ``False``
            - process_group: The process group on which the synchronization is called. Default is the world.
            - dist_sync_fn: function that performs the allgather option on the metric state. Default is an
                custom implementation that calls ``torch.distributed.all_gather`` internally.
            - distributed_available_fn: function that checks if the distributed backend is available.
                Defaults to a check of ``torch.distributed.is_available()`` and ``torch.distributed.is_initialized()``.
            - sync_on_compute: If metric state should synchronize when ``compute`` is called. Default is ``True``-
    """

    __jit_ignored_attributes__ = ["device"]
    __jit_unused_properties__ = ["is_differentiable"]
    is_differentiable: Optional[bool] = None
    higher_is_better: Optional[bool] = None
    full_state_update: Optional[bool] = None

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # see (https://github.com/pytorch/pytorch/blob/3e6bb5233f9ca2c5aa55d9cda22a7ee85439aa6e/
        # torch/nn/modules/module.py#L227)
        torch._C._log_api_usage_once(f"torchmetrics.metric.{self.__class__.__name__}")

        self._device = torch.device("cpu")

        self.compute_on_cpu = kwargs.pop("compute_on_cpu", False)
        if not isinstance(self.compute_on_cpu, bool):
            raise ValueError(
                f"Expected keyword argument `compute_on_cpu` to be an `bool` but got {self.compute_on_cpu}"
            )

        self.dist_sync_on_step = kwargs.pop("dist_sync_on_step", False)
        if not isinstance(self.dist_sync_on_step, bool):
            raise ValueError(
                f"Expected keyword argument `dist_sync_on_step` to be an `bool` but got {self.dist_sync_on_step}"
            )

        self.process_group = kwargs.pop("process_group", None)

        self.dist_sync_fn = kwargs.pop("dist_sync_fn", None)
        if self.dist_sync_fn is not None and not callable(self.dist_sync_fn):
            raise ValueError(
                f"Expected keyword argument `dist_sync_fn` to be an callable function but got {self.dist_sync_fn}"
            )

        self.distributed_available_fn = kwargs.pop("distributed_available_fn", jit_distributed_available)

        self.sync_on_compute = kwargs.pop("sync_on_compute", True)
        if not isinstance(self.sync_on_compute, bool):
            raise ValueError(
                f"Expected keyword argument `sync_on_compute` to be a `bool` but got {self.sync_on_compute}"
            )

        # initialize
        self._update_signature = inspect.signature(self.update)
        self.update: Callable = self._wrap_update(self.update)  # type: ignore
        self.compute: Callable = self._wrap_compute(self.compute)  # type: ignore
        self._computed = None
        self._forward_cache = None
        self._update_count = 0
        self._to_sync = self.sync_on_compute
        self._should_unsync = True
        self._enable_grad = False
        self._dtype_convert = False

        # initialize state
        self._defaults: Dict[str, Union[List, Tensor]] = {}
        self._persistent: Dict[str, bool] = {}
        self._reductions: Dict[str, Union[str, Callable[..., Any], None]] = {}

        # state management
        self._is_synced = False
        self._cache: Optional[Dict[str, Union[List[Tensor], Tensor]]] = None

    @property
    def _update_called(self) -> bool:
        # Needed for lightning integration
        return self._update_count > 0

    def add_state(
        self,
        name: str,
        default: Union[list, Tensor],
        dist_reduce_fx: Optional[Union[str, Callable]] = None,
        persistent: bool = False,
    ) -> None:
        """Adds metric state variable. Only used by subclasses.

        Args:
            name: The name of the state variable. The variable will then be accessible at ``self.name``.
            default: Default value of the state; can either be a :class:`~torch.Tensor` or an empty list.
                The state will be reset to this value when ``self.reset()`` is called.
            dist_reduce_fx (Optional): Function to reduce state across multiple processes in distributed mode.
                If value is ``"sum"``, ``"mean"``, ``"cat"``, ``"min"`` or ``"max"`` we will use ``torch.sum``,
                ``torch.mean``, ``torch.cat``, ``torch.min`` and ``torch.max``` respectively, each with argument
                ``dim=0``. Note that the ``"cat"`` reduction only makes sense if the state is a list, and not
                a tensor. The user can also pass a custom function in this parameter.
            persistent (Optional): whether the state will be saved as part of the modules ``state_dict``.
                Default is ``False``.

        Note:
            Setting ``dist_reduce_fx`` to None will return the metric state synchronized across different processes.
            However, there won't be any reduction function applied to the synchronized metric state.

            The metric states would be synced as follows

            - If the metric state is :class:`~torch.Tensor`, the synced value will be a stacked :class:`~torch.Tensor`
              across the process dimension if the metric state was a :class:`~torch.Tensor`. The original
              :class:`~torch.Tensor` metric state retains dimension and hence the synchronized output will be of shape
              ``(num_process, ...)``.

            - If the metric state is a ``list``, the synced value will be a ``list`` containing the
              combined elements from all processes.

        Note:
            When passing a custom function to ``dist_reduce_fx``, expect the synchronized metric state to follow
            the format discussed in the above note.

        Raises:
            ValueError:
                If ``default`` is not a ``tensor`` or an ``empty list``.
            ValueError:
                If ``dist_reduce_fx`` is not callable or one of ``"mean"``, ``"sum"``, ``"cat"``, ``None``.
        """
        if not isinstance(default, (Tensor, list)) or (isinstance(default, list) and default):
            raise ValueError("state variable must be a tensor or any empty list (where you can append tensors)")

        if dist_reduce_fx == "sum":
            dist_reduce_fx = dim_zero_sum
        elif dist_reduce_fx == "mean":
            dist_reduce_fx = dim_zero_mean
        elif dist_reduce_fx == "max":
            dist_reduce_fx = dim_zero_max
        elif dist_reduce_fx == "min":
            dist_reduce_fx = dim_zero_min
        elif dist_reduce_fx == "cat":
            dist_reduce_fx = dim_zero_cat
        elif dist_reduce_fx is not None and not callable(dist_reduce_fx):
            raise ValueError("`dist_reduce_fx` must be callable or one of ['mean', 'sum', 'cat', None]")

        if isinstance(default, Tensor):
            default = default.contiguous()

        setattr(self, name, default)

        self._defaults[name] = deepcopy(default)
        self._persistent[name] = persistent
        self._reductions[name] = dist_reduce_fx

    @torch.jit.unused
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """``forward`` serves the dual purpose of both computing the metric on the current batch of inputs but also
        add the batch statistics to the overall accumululating metric state.

        Input arguments are the exact same as corresponding ``update`` method. The returned output is the exact same as
        the output of ``compute``.
        """
        # check if states are already synced
        if self._is_synced:
            raise TorchMetricsUserError(
                "The Metric shouldn't be synced when performing ``forward``. "
                "HINT: Did you forget to call ``unsync`` ?."
            )

        if self.full_state_update or self.full_state_update is None or self.dist_sync_on_step:
            self._forward_cache = self._forward_full_state_update(*args, **kwargs)
        else:
            self._forward_cache = self._forward_reduce_state_update(*args, **kwargs)

        return self._forward_cache

    def _forward_full_state_update(self, *args: Any, **kwargs: Any) -> Any:
        """forward computation using two calls to `update` to calculate the metric value on the current batch and
        accumulate global state.

        Doing this secures that metrics that need access to the full metric state during `update` works as expected.
        """
        # global accumulation
        self.update(*args, **kwargs)
        _update_count = self._update_count

        self._to_sync = self.dist_sync_on_step
        # skip restore cache operation from compute as cache is stored below.
        self._should_unsync = False
        # skip computing on cpu for the batch
        _temp_compute_on_cpu = self.compute_on_cpu
        self.compute_on_cpu = False

        # save context before switch
        cache = {attr: getattr(self, attr) for attr in self._defaults}

        # call reset, update, compute, on single batch
        self._enable_grad = True  # allow grads for batch computation
        self.reset()
        self.update(*args, **kwargs)
        batch_val = self.compute()

        # restore context
        for attr, val in cache.items():
            setattr(self, attr, val)
        self._update_count = _update_count

        # restore context
        self._is_synced = False
        self._should_unsync = True
        self._to_sync = self.sync_on_compute
        self._computed = None
        self._enable_grad = False
        self.compute_on_cpu = _temp_compute_on_cpu
        if self.compute_on_cpu:
            self._move_list_states_to_cpu()

        return batch_val

    def _forward_reduce_state_update(self, *args: Any, **kwargs: Any) -> Any:
        """forward computation using single call to `update` to calculate the metric value on the current batch and
        accumulate global state.

        This can be done when the global metric state is a sinple reduction of batch states.
        """
        # store global state and reset to default
        global_state = {attr: getattr(self, attr) for attr in self._defaults.keys()}
        _update_count = self._update_count
        self.reset()

        # local syncronization settings
        self._to_sync = self.dist_sync_on_step
        self._should_unsync = False
        _temp_compute_on_cpu = self.compute_on_cpu
        self.compute_on_cpu = False
        self._enable_grad = True  # allow grads for batch computation

        # calculate batch state and compute batch value
        self.update(*args, **kwargs)
        batch_val = self.compute()

        # reduce batch and global state
        self._update_count = _update_count + 1
        with torch.no_grad():
            self._reduce_states(global_state)

        # restore context
        self._is_synced = False
        self._should_unsync = True
        self._to_sync = self.sync_on_compute
        self._computed = None
        self._enable_grad = False
        self.compute_on_cpu = _temp_compute_on_cpu
        if self.compute_on_cpu:
            self._move_list_states_to_cpu()

        return batch_val

    def _reduce_states(self, incoming_state: Dict[str, Any]) -> None:
        """Adds an incoming metric state to the current state of the metric.

        Args:
            incoming_state: a dict containing a metric state similar metric itself
        """
        for attr in self._defaults.keys():
            local_state = getattr(self, attr)
            global_state = incoming_state[attr]
            reduce_fn = self._reductions[attr]
            if reduce_fn == dim_zero_sum:
                reduced = global_state + local_state
            elif reduce_fn == dim_zero_mean:
                reduced = ((self._update_count - 1) * global_state + local_state).float() / self._update_count
            elif reduce_fn == dim_zero_max:
                reduced = torch.max(global_state, local_state)
            elif reduce_fn == dim_zero_min:
                reduced = torch.min(global_state, local_state)
            elif reduce_fn == dim_zero_cat:
                reduced = global_state + local_state
            elif reduce_fn is None and isinstance(global_state, Tensor):
                reduced = torch.stack([global_state, local_state])
            elif reduce_fn is None and isinstance(global_state, list):
                reduced = _flatten([global_state, local_state])
            else:
                reduced = reduce_fn(torch.stack([global_state, local_state]))  # type: ignore

            setattr(self, attr, reduced)

    def _sync_dist(self, dist_sync_fn: Callable = gather_all_tensors, process_group: Optional[Any] = None) -> None:
        input_dict = {attr: getattr(self, attr) for attr in self._reductions}

        for attr, reduction_fn in self._reductions.items():
            # pre-concatenate metric states that are lists to reduce number of all_gather operations
            if reduction_fn == dim_zero_cat and isinstance(input_dict[attr], list) and len(input_dict[attr]) > 1:
                input_dict[attr] = [dim_zero_cat(input_dict[attr])]

        output_dict = apply_to_collection(
            input_dict,
            Tensor,
            dist_sync_fn,
            group=process_group or self.process_group,
        )

        for attr, reduction_fn in self._reductions.items():
            # pre-processing ops (stack or flatten for inputs)

            if isinstance(output_dict[attr], list) and len(output_dict[attr]) == 0:
                setattr(self, attr, [])
                continue

            if isinstance(output_dict[attr][0], Tensor):
                output_dict[attr] = torch.stack(output_dict[attr])
            elif isinstance(output_dict[attr][0], list):
                output_dict[attr] = _flatten(output_dict[attr])

            if not (callable(reduction_fn) or reduction_fn is None):
                raise TypeError("reduction_fn must be callable or None")
            reduced = reduction_fn(output_dict[attr]) if reduction_fn is not None else output_dict[attr]
            setattr(self, attr, reduced)

    def _wrap_update(self, update: Callable) -> Callable:
        @functools.wraps(update)
        def wrapped_func(*args: Any, **kwargs: Any) -> None:
            self._computed = None
            self._update_count += 1
            with torch.set_grad_enabled(self._enable_grad):
                try:
                    update(*args, **kwargs)
                except RuntimeError as err:
                    if "Expected all tensors to be on" in str(err):
                        raise RuntimeError(
                            "Encountered different devices in metric calculation (see stacktrace for details)."
                            " This could be due to the metric class not being on the same device as input."
                            f" Instead of `metric={self.__class__.__name__}(...)` try to do"
                            f" `metric={self.__class__.__name__}(...).to(device)` where"
                            " device corresponds to the device of the input."
                        ) from err
                    raise err

            if self.compute_on_cpu:
                self._move_list_states_to_cpu()

        return wrapped_func

    def _move_list_states_to_cpu(self) -> None:
        """Move list states to cpu to save GPU memory."""
        for key in self._defaults.keys():
            current_val = getattr(self, key)
            if isinstance(current_val, Sequence):
                setattr(self, key, [cur_v.to("cpu") for cur_v in current_val])

    def sync(
        self,
        dist_sync_fn: Optional[Callable] = None,
        process_group: Optional[Any] = None,
        should_sync: bool = True,
        distributed_available: Optional[Callable] = None,
    ) -> None:
        """Sync function for manually controlling when metrics states should be synced across processes.

        Args:
            dist_sync_fn: Function to be used to perform states synchronization
            process_group:
                Specify the process group on which synchronization is called.
                default: `None` (which selects the entire world)
            should_sync: Whether to apply to state synchronization. This will have an impact
                only when running in a distributed setting.
            distributed_available: Function to determine if we are running inside a distributed setting
        """
        if self._is_synced and should_sync:
            raise TorchMetricsUserError("The Metric has already been synced.")

        if distributed_available is None and self.distributed_available_fn is not None:
            distributed_available = self.distributed_available_fn

        is_distributed = distributed_available() if callable(distributed_available) else None

        if not should_sync or not is_distributed:
            return

        if dist_sync_fn is None:
            dist_sync_fn = gather_all_tensors

        # cache prior to syncing
        self._cache = {attr: getattr(self, attr) for attr in self._defaults}

        # sync
        self._sync_dist(dist_sync_fn, process_group=process_group)
        self._is_synced = True

    def unsync(self, should_unsync: bool = True) -> None:
        """Unsync function for manually controlling when metrics states should be reverted back to their local
        states.

        Args:
            should_unsync: Whether to perform unsync
        """
        if not should_unsync:
            return

        if not self._is_synced:
            raise TorchMetricsUserError("The Metric has already been un-synced.")

        if self._cache is None:
            raise TorchMetricsUserError("The internal cache should exist to unsync the Metric.")

        # if we synced, restore to cache so that we can continue to accumulate un-synced state
        for attr, val in self._cache.items():
            setattr(self, attr, val)
        self._is_synced = False
        self._cache = None

    @contextmanager
    def sync_context(
        self,
        dist_sync_fn: Optional[Callable] = None,
        process_group: Optional[Any] = None,
        should_sync: bool = True,
        should_unsync: bool = True,
        distributed_available: Optional[Callable] = None,
    ) -> Generator:
        """Context manager to synchronize the states between processes when running in a distributed setting and
        restore the local cache states after yielding.

        Args:
            dist_sync_fn: Function to be used to perform states synchronization
            process_group:
                Specify the process group on which synchronization is called.
                default: `None` (which selects the entire world)
            should_sync: Whether to apply to state synchronization. This will have an impact
                only when running in a distributed setting.
            should_unsync: Whether to restore the cache state so that the metrics can
                continue to be accumulated.
            distributed_available: Function to determine if we are running inside a distributed setting
        """
        self.sync(
            dist_sync_fn=dist_sync_fn,
            process_group=process_group,
            should_sync=should_sync,
            distributed_available=distributed_available,
        )

        yield

        self.unsync(should_unsync=self._is_synced and should_unsync)

    def _wrap_compute(self, compute: Callable) -> Callable:
        @functools.wraps(compute)
        def wrapped_func(*args: Any, **kwargs: Any) -> Any:
            if self._update_count == 0:
                rank_zero_warn(
                    f"The ``compute`` method of metric {self.__class__.__name__}"
                    " was called before the ``update`` method which may lead to errors,"
                    " as metric states have not yet been updated.",
                    UserWarning,
                )

            # return cached value
            if self._computed is not None:
                return self._computed

            # compute relies on the sync context manager to gather the states across processes and apply reduction
            # if synchronization happened, the current rank accumulated states will be restored to keep
            # accumulation going if ``should_unsync=True``,
            with self.sync_context(
                dist_sync_fn=self.dist_sync_fn,
                should_sync=self._to_sync,
                should_unsync=self._should_unsync,
            ):
                value = compute(*args, **kwargs)
                self._computed = _squeeze_if_scalar(value)

            return self._computed

        return wrapped_func

    @abstractmethod
    def update(self, *_: Any, **__: Any) -> None:
        """Override this method to update the state variables of your metric class."""

    @abstractmethod
    def compute(self) -> Any:
        """Override this method to compute the final metric value from state variables synchronized across the
        distributed backend."""

    def reset(self) -> None:
        """This method automatically resets the metric state variables to their default value."""
        self._update_count = 0
        self._forward_cache = None
        self._computed = None

        for attr, default in self._defaults.items():
            current_val = getattr(self, attr)
            if isinstance(default, Tensor):
                setattr(self, attr, default.detach().clone().to(current_val.device))
            else:
                setattr(self, attr, [])

        # reset internal states
        self._cache = None
        self._is_synced = False

    def clone(self) -> "Metric":
        """Make a copy of the metric."""
        return deepcopy(self)

    def __getstate__(self) -> Dict[str, Any]:
        # ignore update and compute functions for pickling
        return {k: v for k, v in self.__dict__.items() if k not in ["update", "compute", "_update_signature"]}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        # manually restore update and compute functions for pickling
        self.__dict__.update(state)
        self._update_signature = inspect.signature(self.update)
        self.update: Callable = self._wrap_update(self.update)  # type: ignore
        self.compute: Callable = self._wrap_compute(self.compute)  # type: ignore

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("higher_is_better", "is_differentiable", "full_state_update"):
            raise RuntimeError(f"Can't change const `{name}`.")
        super().__setattr__(name, value)

    @property
    def device(self) -> "torch.device":
        """Return the device of the metric."""
        return self._device

    def type(self, dst_type: Union[str, torch.dtype]) -> "Metric":
        """Method override default and prevent dtype casting.

        Please use `metric.set_dtype(dtype)` instead.
        """
        return self

    def float(self) -> "Metric":
        """Method override default and prevent dtype casting.

        Please use `metric.set_dtype(dtype)` instead.
        """
        return self

    def double(self) -> "Metric":
        """Method override default and prevent dtype casting.

        Please use `metric.set_dtype(dtype)` instead.
        """
        return self

    def half(self) -> "Metric":
        """Method override default and prevent dtype casting.

        Please use `metric.set_dtype(dtype)` instead.
        """
        return self

    def set_dtype(self, dst_type: Union[str, torch.dtype]) -> "Metric":
        """Special version of `type` for transferring all metric states to specific dtype
        Arguments:
            dst_type (type or string): the desired type
        """
        self._dtype_convert = True
        out = super().type(dst_type)
        out._dtype_convert = False
        return out

    def _apply(self, fn: Callable) -> Module:
        """Overwrite _apply function such that we can also move metric states to the correct device.

        This method is called by the base ``nn.Module`` class whenever `.to`, `.cuda`, `.float`, `.half` etc. methods
        are called. Dtype conversion is garded and will only happen through the special `set_dtype` method.
        """
        this = super()._apply(fn)
        fs = str(fn)
        cond = any(f in fs for f in ["Module.type", "Module.half", "Module.float", "Module.double", "Module.bfloat16"])
        if not self._dtype_convert and cond:
            return this

        # Also apply fn to metric states and defaults
        for key, value in this._defaults.items():
            if isinstance(value, Tensor):
                this._defaults[key] = fn(value)
            elif isinstance(value, Sequence):
                this._defaults[key] = [fn(v) for v in value]

            current_val = getattr(this, key)
            if isinstance(current_val, Tensor):
                setattr(this, key, fn(current_val))
            elif isinstance(current_val, Sequence):
                setattr(this, key, [fn(cur_v) for cur_v in current_val])
            else:
                raise TypeError(
                    "Expected metric state to be either a Tensor" f"or a list of Tensor, but encountered {current_val}"
                )

        # make sure to update the device attribute
        # if the dummy tensor moves device by fn function we should also update the attribute
        self._device = fn(torch.zeros(1, device=self.device)).device

        # Additional apply to forward cache and computed attributes (may be nested)
        if this._computed is not None:
            this._computed = apply_to_collection(this._computed, Tensor, fn)
        if this._forward_cache is not None:
            this._forward_cache = apply_to_collection(this._forward_cache, Tensor, fn)

        return this

    def persistent(self, mode: bool = False) -> None:
        """Method for post-init to change if metric states should be saved to its state_dict."""
        for key in self._persistent:
            self._persistent[key] = mode

    def state_dict(
        self,
        destination: Dict[str, Any] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Optional[Dict[str, Any]]:
        destination = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        # Register metric states to be part of the state_dict
        for key in self._defaults:
            if not self._persistent[key]:
                continue
            current_val = getattr(self, key)
            if not keep_vars:
                if isinstance(current_val, Tensor):
                    current_val = current_val.detach()
                elif isinstance(current_val, list):
                    current_val = [cur_v.detach() if isinstance(cur_v, Tensor) else cur_v for cur_v in current_val]
            destination[prefix + key] = deepcopy(current_val)  # type: ignore
        return destination

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        """Loads metric states from state_dict."""

        for key in self._defaults:
            name = prefix + key
            if name in state_dict:
                setattr(self, key, state_dict.pop(name))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
        )

    def _filter_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        """filter kwargs such that they match the update signature of the metric."""

        # filter all parameters based on update signature except those of
        # type VAR_POSITIONAL (*args) and VAR_KEYWORD (**kwargs)
        _params = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        _sign_params = self._update_signature.parameters
        filtered_kwargs = {
            k: v for k, v in kwargs.items() if (k in _sign_params.keys() and _sign_params[k].kind not in _params)
        }

        exists_var_keyword = any(v.kind == inspect.Parameter.VAR_KEYWORD for v in _sign_params.values())
        # if no kwargs filtered, return all kwargs as default
        if not filtered_kwargs and not exists_var_keyword:
            # no kwargs in update signature -> don't return any kwargs
            filtered_kwargs = {}
        elif exists_var_keyword:
            # kwargs found in update signature -> return all kwargs to be sure to not omit any.
            # filtering logic is likely implemented within the update call.
            filtered_kwargs = kwargs
        return filtered_kwargs

    def __hash__(self) -> int:
        # we need to add the id here, since PyTorch requires a module hash to be unique.
        # Internally, PyTorch nn.Module relies on that for children discovery
        # (see https://github.com/pytorch/pytorch/blob/v1.9.0/torch/nn/modules/module.py#L1544)
        # For metrics that include tensors it is not a problem,
        # since their hash is unique based on the memory location but we cannot rely on that for every metric.
        hash_vals = [self.__class__.__name__, id(self)]

        for key in self._defaults:
            val = getattr(self, key)
            # Special case: allow list values, so long
            # as their elements are hashable
            if hasattr(val, "__iter__") and not isinstance(val, Tensor):
                hash_vals.extend(val)
            else:
                hash_vals.append(val)

        return hash(tuple(hash_vals))

    def __add__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.add, self, other)

    def __and__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.bitwise_and, self, other)

    # Fixme: this shall return bool instead of Metric
    def __eq__(self, other: "Metric") -> "Metric":  # type: ignore
        return CompositionalMetric(torch.eq, self, other)

    def __floordiv__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.floor_divide, self, other)

    def __ge__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.ge, self, other)

    def __gt__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.gt, self, other)

    def __le__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.le, self, other)

    def __lt__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.lt, self, other)

    def __matmul__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.matmul, self, other)

    def __mod__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.fmod, self, other)

    def __mul__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.mul, self, other)

    # Fixme: this shall return bool instead of Metric
    def __ne__(self, other: "Metric") -> "Metric":  # type: ignore
        return CompositionalMetric(torch.ne, self, other)

    def __or__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.bitwise_or, self, other)

    def __pow__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.pow, self, other)

    def __radd__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.add, other, self)

    def __rand__(self, other: "Metric") -> "Metric":
        # swap them since bitwise_and only supports that way and it's commutative
        return CompositionalMetric(torch.bitwise_and, self, other)

    def __rfloordiv__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.floor_divide, other, self)

    def __rmatmul__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.matmul, other, self)

    def __rmod__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.fmod, other, self)

    def __rmul__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.mul, other, self)

    def __ror__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.bitwise_or, other, self)

    def __rpow__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.pow, other, self)

    def __rsub__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.sub, other, self)

    def __rtruediv__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.true_divide, other, self)

    def __rxor__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.bitwise_xor, other, self)

    def __sub__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.sub, self, other)

    def __truediv__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.true_divide, self, other)

    def __xor__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.bitwise_xor, self, other)

    def __abs__(self) -> "Metric":
        return CompositionalMetric(torch.abs, self, None)

    def __inv__(self) -> "Metric":
        return CompositionalMetric(torch.bitwise_not, self, None)

    def __invert__(self) -> "Metric":
        return self.__inv__()

    def __neg__(self) -> "Metric":
        return CompositionalMetric(_neg, self, None)

    def __pos__(self) -> "Metric":
        return CompositionalMetric(torch.abs, self, None)

    def __getitem__(self, idx: int) -> "Metric":
        return CompositionalMetric(lambda x: x[idx], self, None)

    def __getnewargs__(self) -> Tuple:
        return (Metric.__str__(self),)

    def __iter__(self):
        raise NotImplementedError("Metrics does not support iteration.")


def _neg(x: Tensor) -> Tensor:
    return -torch.abs(x)


class CompositionalMetric(Metric):
    """Composition of two metrics with a specific operator which will be executed upon metrics compute."""

    def __init__(
        self,
        operator: Callable,
        metric_a: Union[Metric, int, float, Tensor],
        metric_b: Union[Metric, int, float, Tensor, None],
    ) -> None:
        """
        Args:
            operator: the operator taking in one (if metric_b is None)
                or two arguments. Will be applied to outputs of metric_a.compute()
                and (optionally if metric_b is not None) metric_b.compute()
            metric_a: first metric whose compute() result is the first argument of operator
            metric_b: second metric whose compute() result is the second argument of operator.
                For operators taking in only one input, this should be None
        """
        super().__init__()

        self.op = operator

        if isinstance(metric_a, Tensor):
            self.register_buffer("metric_a", metric_a)
        else:
            self.metric_a = metric_a

        if isinstance(metric_b, Tensor):
            self.register_buffer("metric_b", metric_b)
        else:
            self.metric_b = metric_b

    def _sync_dist(self, dist_sync_fn: Optional[Callable] = None, process_group: Optional[Any] = None) -> None:
        # No syncing required here. syncing will be done in metric_a and metric_b
        pass

    def update(self, *args: Any, **kwargs: Any) -> None:
        if isinstance(self.metric_a, Metric):
            self.metric_a.update(*args, **self.metric_a._filter_kwargs(**kwargs))

        if isinstance(self.metric_b, Metric):
            self.metric_b.update(*args, **self.metric_b._filter_kwargs(**kwargs))

    def compute(self) -> Any:
        # also some parsing for kwargs?
        if isinstance(self.metric_a, Metric):
            val_a = self.metric_a.compute()
        else:
            val_a = self.metric_a

        if isinstance(self.metric_b, Metric):
            val_b = self.metric_b.compute()
        else:
            val_b = self.metric_b

        if val_b is None:
            return self.op(val_a)

        return self.op(val_a, val_b)

    @torch.jit.unused
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        val_a = (
            self.metric_a(*args, **self.metric_a._filter_kwargs(**kwargs))
            if isinstance(self.metric_a, Metric)
            else self.metric_a
        )
        val_b = (
            self.metric_b(*args, **self.metric_b._filter_kwargs(**kwargs))
            if isinstance(self.metric_b, Metric)
            else self.metric_b
        )

        if val_a is None:
            return None

        if val_b is None:
            if isinstance(self.metric_b, Metric):
                return None

            # Unary op
            return self.op(val_a)

        # Binary op
        return self.op(val_a, val_b)

    def reset(self) -> None:
        if isinstance(self.metric_a, Metric):
            self.metric_a.reset()

        if isinstance(self.metric_b, Metric):
            self.metric_b.reset()

    def persistent(self, mode: bool = False) -> None:
        if isinstance(self.metric_a, Metric):
            self.metric_a.persistent(mode=mode)
        if isinstance(self.metric_b, Metric):
            self.metric_b.persistent(mode=mode)

    def __repr__(self) -> str:
        _op_metrics = f"(\n  {self.op.__name__}(\n    {repr(self.metric_a)},\n    {repr(self.metric_b)}\n  )\n)"
        repr_str = self.__class__.__name__ + _op_metrics

        return repr_str

    def _wrap_compute(self, compute: Callable) -> Callable:
        return compute

def rank_zero_only(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapped_fn(*args: Any, **kwargs: Any) -> Any:
        if rank_zero_only.rank == 0:  # type: ignore
            return fn(*args, **kwargs)

    return wrapped_fn

rank_zero_warn = rank_zero_only(_warn)
class MetricCollection(ModuleDict):
    """MetricCollection class can be used to chain metrics that have the same call pattern into one single class.

    Args:
        metrics: One of the following

            * list or tuple (sequence): if metrics are passed in as a list or tuple, will use the metrics class name
              as key for output dict. Therefore, two metrics of the same class cannot be chained this way.

            * arguments: similar to passing in as a list, metrics passed in as arguments will use their metric
              class name as key for the output dict.

            * dict: if metrics are passed in as a dict, will use each key in the dict as key for output dict.
              Use this format if you want to chain together multiple of the same metric with different parameters.
              Note that the keys in the output dict will be sorted alphabetically.

        prefix: a string to append in front of the keys of the output dict

        postfix: a string to append after the keys of the output dict

        compute_groups:
            By default the MetricCollection will try to reduce the computations needed for the metrics in the collection
            by checking if they belong to the same **compute group**. All metrics in a compute group share the same
            metric state and are therefore only different in their compute step e.g. accuracy, precision and recall
            can all be computed from the true positives/negatives and false positives/negatives. By default,
            this argument is ``True`` which enables this feature. Set this argument to `False` for disabling
            this behaviour. Can also be set to a list of lists of metrics for setting the compute groups yourself.

    .. note::
        The compute groups feature can significatly speedup the calculation of metrics under the right conditions.
        First, the feature is only available when calling the ``update`` method and not when calling ``forward`` method
        due to the internal logic of ``forward`` preventing this. Secondly, since we compute groups share metric
        states by reference, calling ``.items()``, ``.values()`` etc. on the metric collection will break this
        reference and a copy of states are instead returned in this case (reference will be reestablished on the next
        call to ``update``).

    .. note::
        Metric collections can be nested at initilization (see last example) but the output of the collection will
        still be a single flatten dictionary combining the prefix and postfix arguments from the nested collection.

    Raises:
        ValueError:
            If one of the elements of ``metrics`` is not an instance of ``pl.metrics.Metric``.
        ValueError:
            If two elements in ``metrics`` have the same ``name``.
        ValueError:
            If ``metrics`` is not a ``list``, ``tuple`` or a ``dict``.
        ValueError:
            If ``metrics`` is ``dict`` and additional_metrics are passed in.
        ValueError:
            If ``prefix`` is set and it is not a string.
        ValueError:
            If ``postfix`` is set and it is not a string.

    Example (input as list):
        >>> import torch
        >>> from pprint import pprint
        >>> from torchmetrics import MetricCollection, MeanSquaredError
        >>> from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall
        >>> target = torch.tensor([0, 2, 0, 2, 0, 1, 0, 2])
        >>> preds = torch.tensor([2, 1, 2, 0, 1, 2, 2, 2])
        >>> metrics = MetricCollection([MulticlassAccuracy(num_classes=3, average='micro'),
        ...                             MulticlassPrecision(num_classes=3, average='macro'),
        ...                             MulticlassRecall(num_classes=3, average='macro')])
        >>> metrics(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        {'MulticlassAccuracy': tensor(0.1250),
         'MulticlassPrecision': tensor(0.0667),
         'MulticlassRecall': tensor(0.1111)}

    Example (input as arguments):
        >>> metrics = MetricCollection(MulticlassAccuracy(num_classes=3, average='micro'),
        ...                            MulticlassPrecision(num_classes=3, average='macro'),
        ...                            MulticlassRecall(num_classes=3, average='macro'))
        >>> metrics(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        {'MulticlassAccuracy': tensor(0.1250),
         'MulticlassPrecision': tensor(0.0667),
         'MulticlassRecall': tensor(0.1111)}

    Example (input as dict):
        >>> metrics = MetricCollection({'micro_recall': MulticlassRecall(num_classes=3, average='micro'),
        ...                             'macro_recall': MulticlassRecall(num_classes=3, average='macro')})
        >>> same_metric = metrics.clone()
        >>> pprint(metrics(preds, target))
        {'macro_recall': tensor(0.1111), 'micro_recall': tensor(0.1250)}
        >>> pprint(same_metric(preds, target))
        {'macro_recall': tensor(0.1111), 'micro_recall': tensor(0.1250)}

    Example (specification of compute groups):
        >>> metrics = MetricCollection(
        ...     MulticlassRecall(num_classes=3, average='macro'),
        ...     MulticlassPrecision(num_classes=3, average='macro'),
        ...     MeanSquaredError(),
        ...     compute_groups=[['MulticlassRecall', 'MulticlassPrecision'], ['MeanSquaredError']]
        ... )
        >>> metrics.update(preds, target)
        >>> pprint(metrics.compute())
        {'MeanSquaredError': tensor(2.3750), 'MulticlassPrecision': tensor(0.0667), 'MulticlassRecall': tensor(0.1111)}
        >>> pprint(metrics.compute_groups)
        {0: ['MulticlassRecall', 'MulticlassPrecision'], 1: ['MeanSquaredError']}

    Example (nested metric collections):
        >>> metrics = MetricCollection([
        ...     MetricCollection([
        ...         MulticlassAccuracy(num_classes=3, average='macro'),
        ...         MulticlassPrecision(num_classes=3, average='macro')
        ...     ], postfix='_macro'),
        ...     MetricCollection([
        ...         MulticlassAccuracy(num_classes=3, average='micro'),
        ...         MulticlassPrecision(num_classes=3, average='micro')
        ...     ], postfix='_micro'),
        ... ], prefix='valmetrics/')
        >>> pprint(metrics(preds, target))  # doctest: +NORMALIZE_WHITESPACE
        {'valmetrics/MulticlassAccuracy_macro': tensor(0.1111),
         'valmetrics/MulticlassAccuracy_micro': tensor(0.1250),
         'valmetrics/MulticlassPrecision_macro': tensor(0.0667),
         'valmetrics/MulticlassPrecision_micro': tensor(0.1250)}
    """

    _groups: Dict[int, List[str]]

    def __init__(
        self,
        metrics: Union[Metric, Sequence[Metric], Dict[str, Metric]],
        *additional_metrics: Metric,
        prefix: Optional[str] = None,
        postfix: Optional[str] = None,
        compute_groups: Union[bool, List[List[str]]] = True,
    ) -> None:
        super().__init__()

        self.prefix = self._check_arg(prefix, "prefix")
        self.postfix = self._check_arg(postfix, "postfix")
        self._enable_compute_groups = compute_groups
        self._groups_checked: bool = False
        self._state_is_copy: bool = False

        self.add_metrics(metrics, *additional_metrics)

    @torch.jit.unused
    def forward(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Iteratively call forward for each metric.

        Positional arguments (args) will be passed to every metric in the collection, while keyword arguments (kwargs)
        will be filtered based on the signature of the individual metric.
        """
        res = {k: m(*args, **m._filter_kwargs(**kwargs)) for k, m in self.items(keep_base=True, copy_state=False)}
        res = _flatten_dict(res)
        return {self._set_name(k): v for k, v in res.items()}

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Iteratively call update for each metric.

        Positional arguments (args) will be passed to every metric in the collection, while keyword arguments (kwargs)
        will be filtered based on the signature of the individual metric.
        """
        # Use compute groups if already initialized and checked
        if self._groups_checked:
            for _, cg in self._groups.items():
                # only update the first member
                m0 = getattr(self, cg[0])
                m0.update(*args, **m0._filter_kwargs(**kwargs))
            if self._state_is_copy:
                # If we have deep copied state inbetween updates, reestablish link
                self._compute_groups_create_state_ref()
                self._state_is_copy = False
        else:  # the first update always do per metric to form compute groups
            for _, m in self.items(keep_base=True, copy_state=False):
                m_kwargs = m._filter_kwargs(**kwargs)
                m.update(*args, **m_kwargs)

            if self._enable_compute_groups:
                self._merge_compute_groups()
                # create reference between states
                self._compute_groups_create_state_ref()
                self._groups_checked = True

    def _merge_compute_groups(self) -> None:
        """Iterates over the collection of metrics, checking if the state of each metric matches another.

        If so, their compute groups will be merged into one. The complexity of the method is approximately
        ``O(number_of_metrics_in_collection ** 2)``, as all metrics need to be compared to all other metrics.
        """
        n_groups = len(self._groups)
        while True:
            for cg_idx1, cg_members1 in deepcopy(self._groups).items():
                for cg_idx2, cg_members2 in deepcopy(self._groups).items():
                    if cg_idx1 == cg_idx2:
                        continue

                    metric1 = getattr(self, cg_members1[0])
                    metric2 = getattr(self, cg_members2[0])

                    if self._equal_metric_states(metric1, metric2):
                        self._groups[cg_idx1].extend(self._groups.pop(cg_idx2))
                        break

                # Start over if we merged groups
                if len(self._groups) != n_groups:
                    break

            # Stop when we iterate over everything and do not merge any groups
            if len(self._groups) == n_groups:
                break
            else:
                n_groups = len(self._groups)

        # Re-index groups
        temp = deepcopy(self._groups)
        self._groups = {}
        for idx, values in enumerate(temp.values()):
            self._groups[idx] = values

    @staticmethod
    def _equal_metric_states(metric1: Metric, metric2: Metric) -> bool:
        """Check if the metric state of two metrics are the same."""
        # empty state
        if len(metric1._defaults) == 0 or len(metric2._defaults) == 0:
            return False

        if metric1._defaults.keys() != metric2._defaults.keys():
            return False

        for key in metric1._defaults.keys():
            state1 = getattr(metric1, key)
            state2 = getattr(metric2, key)

            if type(state1) != type(state2):
                return False

            if isinstance(state1, Tensor) and isinstance(state2, Tensor):
                return state1.shape == state2.shape and allclose(state1, state2)

            if isinstance(state1, list) and isinstance(state2, list):
                return all(s1.shape == s2.shape and allclose(s1, s2) for s1, s2 in zip(state1, state2))

        return True

    def _compute_groups_create_state_ref(self, copy: bool = False) -> None:
        """Create reference between metrics in the same compute group.

        Args:
            copy: If `True` the metric state will between members will be copied instead
                of just passed by reference
        """
        if not self._state_is_copy:
            for _, cg in self._groups.items():
                m0 = getattr(self, cg[0])
                for i in range(1, len(cg)):
                    mi = getattr(self, cg[i])
                    for state in m0._defaults:
                        m0_state = getattr(m0, state)
                        # Determine if we just should set a reference or a full copy
                        setattr(mi, state, deepcopy(m0_state) if copy else m0_state)
                    setattr(mi, "_update_count", deepcopy(m0._update_count) if copy else m0._update_count)
        self._state_is_copy = copy

    def compute(self) -> Dict[str, Any]:
        """Compute the result for each metric in the collection."""
        res = {k: m.compute() for k, m in self.items(keep_base=True, copy_state=False)}
        res = _flatten_dict(res)
        return {self._set_name(k): v for k, v in res.items()}

    def reset(self) -> None:
        """Iteratively call reset for each metric."""
        for _, m in self.items(keep_base=True, copy_state=False):
            m.reset()
        if self._enable_compute_groups and self._groups_checked:
            # reset state reference
            self._compute_groups_create_state_ref()

    def clone(self, prefix: Optional[str] = None, postfix: Optional[str] = None) -> "MetricCollection":
        """Make a copy of the metric collection
        Args:
            prefix: a string to append in front of the metric keys
            postfix: a string to append after the keys of the output dict

        """
        mc = deepcopy(self)
        if prefix:
            mc.prefix = self._check_arg(prefix, "prefix")
        if postfix:
            mc.postfix = self._check_arg(postfix, "postfix")
        return mc

    def persistent(self, mode: bool = True) -> None:
        """Method for post-init to change if metric states should be saved to its state_dict."""
        for _, m in self.items(keep_base=True, copy_state=False):
            m.persistent(mode)

    def add_metrics(
        self, metrics: Union[Metric, Sequence[Metric], Dict[str, Metric]], *additional_metrics: Metric
    ) -> None:
        """Add new metrics to Metric Collection."""
        if isinstance(metrics, Metric):
            # set compatible with original type expectations
            metrics = [metrics]
        if isinstance(metrics, Sequence):
            # prepare for optional additions
            metrics = list(metrics)
            remain: list = []
            for m in additional_metrics:
                (metrics if isinstance(m, Metric) else remain).append(m)

            if remain:
                rank_zero_warn(
                    f"You have passes extra arguments {remain} which are not `Metric` so they will be ignored."
                )
        elif additional_metrics:
            raise ValueError(
                f"You have passes extra arguments {additional_metrics} which are not compatible"
                f" with first passed dictionary {metrics} so they will be ignored."
            )

        if isinstance(metrics, dict):
            # Check all values are metrics
            # Make sure that metrics are added in deterministic order
            for name in sorted(metrics.keys()):
                metric = metrics[name]
                if not isinstance(metric, (Metric, MetricCollection)):
                    raise ValueError(
                        f"Value {metric} belonging to key {name} is not an instance of"
                        " `torchmetrics.Metric` or `torchmetrics.MetricCollection`"
                    )
                if isinstance(metric, Metric):
                    self[name] = metric
                else:
                    for k, v in metric.items(keep_base=False):
                        self[f"{name}_{k}"] = v
        elif isinstance(metrics, Sequence):
            for metric in metrics:
                if not isinstance(metric, (Metric, MetricCollection)):
                    raise ValueError(
                        f"Input {metric} to `MetricCollection` is not a instance of"
                        " `torchmetrics.Metric` or `torchmetrics.MetricCollection`"
                    )
                if isinstance(metric, Metric):
                    name = metric.__class__.__name__
                    if name in self:
                        raise ValueError(f"Encountered two metrics both named {name}")
                    self[name] = metric
                else:
                    for k, v in metric.items(keep_base=False):
                        self[k] = v
        else:
            raise ValueError("Unknown input to MetricCollection.")

        self._groups_checked = False
        if self._enable_compute_groups:
            self._init_compute_groups()
        else:
            self._groups = {}

    def _init_compute_groups(self) -> None:
        """Initialize compute groups.

        If user provided a list, we check that all metrics in the list are also in the collection. If set to `True` we
        simply initialize each metric in the collection as its own group
        """
        if isinstance(self._enable_compute_groups, list):
            self._groups = {i: k for i, k in enumerate(self._enable_compute_groups)}
            for v in self._groups.values():
                for metric in v:
                    if metric not in self:
                        raise ValueError(
                            f"Input {metric} in `compute_groups` argument does not match a metric in the collection."
                            f" Please make sure that {self._enable_compute_groups} matches {self.keys(keep_base=True)}"
                        )
            self._groups_checked = True
        else:
            # Initialize all metrics as their own compute group
            self._groups = {i: [str(k)] for i, k in enumerate(self.keys(keep_base=True))}

    @property
    def compute_groups(self) -> Dict[int, List[str]]:
        """Return a dict with the current compute groups in the collection."""
        return self._groups

    def _set_name(self, base: str) -> str:
        """Adjust name of metric with both prefix and postfix."""
        name = base if self.prefix is None else self.prefix + base
        name = name if self.postfix is None else name + self.postfix
        return name

    def _to_renamed_ordered_dict(self) -> OrderedDict:
        od = OrderedDict()
        for k, v in self._modules.items():
            od[self._set_name(k)] = v
        return od

    def keys(self, keep_base: bool = False) -> Iterable[Hashable]:
        r"""Return an iterable of the ModuleDict key.

        Args:
            keep_base: Whether to add prefix/postfix on the items collection.
        """
        if keep_base:
            return self._modules.keys()
        return self._to_renamed_ordered_dict().keys()

    def items(self, keep_base: bool = False, copy_state: bool = True) -> Iterable[Tuple[str, Module]]:
        r"""Return an iterable of the ModuleDict key/value pairs.

        Args:
            keep_base: Whether to add prefix/postfix on the collection.
            copy_state:
                If metric states should be copied between metrics in the same compute group or just passed by reference
        """
        self._compute_groups_create_state_ref(copy_state)
        if keep_base:
            return self._modules.items()
        return self._to_renamed_ordered_dict().items()

    def values(self, copy_state: bool = True) -> Iterable[Module]:
        """Return an iterable of the ModuleDict values.

        Args:
            copy_state:
                If metric states should be copied between metrics in the same compute group or just passed by reference
        """
        self._compute_groups_create_state_ref(copy_state)
        return self._modules.values()

    def __getitem__(self, key: str, copy_state: bool = True) -> Module:
        """Retrieve a single metric from the collection.

        Args:
            key: name of metric to retrieve
            copy_state:
                If metric states should be copied between metrics in the same compute group or just passed by reference
        """
        self._compute_groups_create_state_ref(copy_state)
        return self._modules[key]

    @staticmethod
    def _check_arg(arg: Optional[str], name: str) -> Optional[str]:
        if arg is None or isinstance(arg, str):
            return arg
        raise ValueError(f"Expected input `{name}` to be a string, but got {type(arg)}")

    def __repr__(self) -> str:
        repr_str = super().__repr__()[:-2]
        if self.prefix:
            repr_str += f",\n  prefix={self.prefix}{',' if self.postfix else ''}"
        if self.postfix:
            repr_str += f"{',' if not self.prefix else ''}\n  postfix={self.postfix}"
        return repr_str + "\n)"

    def set_dtype(self, dst_type: Union[str, torch.dtype]) -> "MetricCollection":
        """Transfer all metric state to specific dtype. Special version of standard `type` method.

        Arguments:
            dst_type (type or string): the desired type.
        """
        for _, m in self.items(keep_base=True, copy_state=False):
            m.set_dtype(dst_type)
        return self

def _accuracy_reduce(
    tp: Tensor,
    fp: Tensor,
    tn: Tensor,
    fn: Tensor,
    average: Optional[Literal["binary", "micro", "macro", "weighted", "none"]],
    multidim_average: Literal["global", "samplewise"] = "global",
    multilabel: bool = False,
) -> Tensor:
    """Reduce classification statistics into accuracy score
    Args:
        tp: number of true positives
        fp: number of false positives
        tn: number of true negatives
        fn: number of false negatives
        normalize: normalization method.
            - `"true"` will divide by the sum of the column dimension.
            - `"pred"` will divide by the sum of the row dimension.
            - `"all"` will divide by the sum of the full matrix
            - `"none"` or `None` will apply no reduction
        multilabel: bool indicating if reduction is for multilabel tasks

    Returns:
        Accuracy score
    """
    if average == "binary":
        return _safe_divide(tp + tn, tp + tn + fp + fn)
    elif average == "micro":
        tp = tp.sum(dim=0 if multidim_average == "global" else 1)
        fn = fn.sum(dim=0 if multidim_average == "global" else 1)
        if multilabel:
            fp = fp.sum(dim=0 if multidim_average == "global" else 1)
            tn = tn.sum(dim=0 if multidim_average == "global" else 1)
            return _safe_divide(tp + tn, tp + tn + fp + fn)
        return _safe_divide(tp, tp + fn)
    else:
        if multilabel:
            score = _safe_divide(tp + tn, tp + tn + fp + fn)
        else:
            score = _safe_divide(tp, tp + fn)
        if average is None or average == "none":
            return score
        if average == "weighted":
            weights = tp + fn
        else:
            weights = torch.ones_like(score)
        return _safe_divide(weights * score, weights.sum(-1, keepdim=True)).sum(-1)

class _AbstractStatScores(Metric):
    # define common functions
    def _create_state(
        self,
        size: int,
        multidim_average: Literal["global", "samplewise"] = "global",
    ) -> None:
        """Initialize the states for the different statistics."""
        default: Union[Callable[[], list], Callable[[], Tensor]]
        if multidim_average == "samplewise":
            default = lambda: []
            dist_reduce_fx = "cat"
        else:
            default = lambda: torch.zeros(size, dtype=torch.long)
            dist_reduce_fx = "sum"

        self.add_state("tp", default(), dist_reduce_fx=dist_reduce_fx)
        self.add_state("fp", default(), dist_reduce_fx=dist_reduce_fx)
        self.add_state("tn", default(), dist_reduce_fx=dist_reduce_fx)
        self.add_state("fn", default(), dist_reduce_fx=dist_reduce_fx)

    def _update_state(self, tp: Tensor, fp: Tensor, tn: Tensor, fn: Tensor) -> None:
        """Update states depending on multidim_average argument."""
        if self.multidim_average == "samplewise":
            self.tp.append(tp)
            self.fp.append(fp)
            self.tn.append(tn)
            self.fn.append(fn)
        else:
            self.tp += tp
            self.fp += fp
            self.tn += tn
            self.fn += fn

    def _final_state(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Final aggregation in case of list states."""
        tp = dim_zero_cat(self.tp)
        fp = dim_zero_cat(self.fp)
        tn = dim_zero_cat(self.tn)
        fn = dim_zero_cat(self.fn)
        return tp, fp, tn, fn
    
class MulticlassStatScores(_AbstractStatScores):
    r"""Computes the number of true positives, false positives, true negatives, false negatives and the support for
    multiclass tasks. Related to `Type I and Type II errors`_.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, ...)`` or float tensor of shape ``(N, C, ..)``.
      If preds is a floating point we apply ``torch.argmax`` along the ``C`` dimension to automatically convert
      probabilities/logits into an int tensor.
    - ``target`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, ...)``


    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``mcss`` (:class:`~torch.Tensor`): A tensor of shape ``(..., 5)``, where the last dimension corresponds
      to ``[tp, fp, tn, fn, sup]`` (``sup`` stands for support and equals ``tp + fn``). The shape
      depends on ``average`` and ``multidim_average`` parameters:

    - If ``multidim_average`` is set to ``global``
    - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(5,)``
    - If ``average=None/'none'``, the shape will be ``(C, 5)``
    - If ``multidim_average`` is set to ``samplewise``
    - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(N, 5)``
    - If ``average=None/'none'``, the shape will be ``(N, C, 5)``

    Args:
        num_classes: Integer specifing the number of classes
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: Calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: Calculates statistic for each label and applies no reduction
        top_k:
            Number of highest probability or logit score predictions considered to find the correct label.
            Only works when ``preds`` contain probabilities/logits.
        multidim_average:
            Defines how additionally dimensions ``...`` should be handled. Should be one of the following:

            - ``global``: Additional dimensions are flatted along the batch dimension
            - ``samplewise``: Statistic will be calculated independently for each sample on the ``N`` axis.
              The statistics in this case are calculated over the additional dimensions.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (preds is int tensor):
        >>> from torchmetrics.classification import MulticlassStatScores
        >>> target = torch.tensor([2, 1, 0, 0])
        >>> preds = torch.tensor([2, 1, 0, 1])
        >>> metric = MulticlassStatScores(num_classes=3, average='micro')
        >>> metric(preds, target)
        tensor([3, 1, 7, 1, 4])
        >>> mcss = MulticlassStatScores(num_classes=3, average=None)
        >>> mcss(preds, target)
        tensor([[1, 0, 2, 1, 2],
                [1, 1, 2, 0, 1],
                [1, 0, 3, 0, 1]])

    Example (preds is float tensor):
        >>> from torchmetrics.classification import MulticlassStatScores
        >>> target = torch.tensor([2, 1, 0, 0])
        >>> preds = torch.tensor([
        ...   [0.16, 0.26, 0.58],
        ...   [0.22, 0.61, 0.17],
        ...   [0.71, 0.09, 0.20],
        ...   [0.05, 0.82, 0.13],
        ... ])
        >>> metric = MulticlassStatScores(num_classes=3, average='micro')
        >>> metric(preds, target)
        tensor([3, 1, 7, 1, 4])
        >>> mcss = MulticlassStatScores(num_classes=3, average=None)
        >>> mcss(preds, target)
        tensor([[1, 0, 2, 1, 2],
                [1, 1, 2, 0, 1],
                [1, 0, 3, 0, 1]])

    Example (multidim tensors):
        >>> from torchmetrics.classification import MulticlassStatScores
        >>> target = torch.tensor([[[0, 1], [2, 1], [0, 2]], [[1, 1], [2, 0], [1, 2]]])
        >>> preds = torch.tensor([[[0, 2], [2, 0], [0, 1]], [[2, 2], [2, 1], [1, 0]]])
        >>> metric = MulticlassStatScores(num_classes=3, multidim_average="samplewise", average='micro')
        >>> metric(preds, target)
        tensor([[3, 3, 9, 3, 6],
                [2, 4, 8, 4, 6]])
        >>> mcss = MulticlassStatScores(num_classes=3, multidim_average="samplewise", average=None)
        >>> mcss(preds, target)
        tensor([[[2, 1, 3, 0, 2],
                 [0, 1, 3, 2, 2],
                 [1, 1, 3, 1, 2]],
                [[0, 1, 4, 1, 1],
                 [1, 1, 2, 2, 3],
                 [1, 2, 2, 1, 2]]])
    """
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: int,
        top_k: int = 1,
        average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
        multidim_average: Literal["global", "samplewise"] = "global",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super(_AbstractStatScores, self).__init__(**kwargs)
        if validate_args:
            _multiclass_stat_scores_arg_validation(num_classes, top_k, average, multidim_average, ignore_index)
        self.num_classes = num_classes
        self.top_k = top_k
        self.average = average
        self.multidim_average = multidim_average
        self.ignore_index = ignore_index
        self.validate_args = validate_args

        self._create_state(
            size=1 if (average == "micro" and top_k == 1) else num_classes, multidim_average=multidim_average
        )

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets."""
        if self.validate_args:
            _multiclass_stat_scores_tensor_validation(
                preds, target, self.num_classes, self.multidim_average, self.ignore_index
            )
        preds, target = _multiclass_stat_scores_format(preds, target, self.top_k)
        tp, fp, tn, fn = _multiclass_stat_scores_update(
            preds, target, self.num_classes, self.top_k, self.average, self.multidim_average, self.ignore_index
        )
        self._update_state(tp, fp, tn, fn)

    def compute(self) -> Tensor:
        """Computes the final statistics."""
        tp, fp, tn, fn = self._final_state()
        return _multiclass_stat_scores_compute(tp, fp, tn, fn, self.average, self.multidim_average)
    
class MulticlassAccuracy(MulticlassStatScores):
    r"""Computes `Accuracy`_ for multiclass tasks:

    .. math::
        \text{Accuracy} = \frac{1}{N}\sum_i^N 1(y_i = \hat{y}_i)

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a tensor of predictions.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, ...)`` or float tensor of shape ``(N, C, ..)``.
      If preds is a floating point we apply ``torch.argmax`` along the ``C`` dimension to automatically convert
      probabilities/logits into an int tensor.
    - ``target`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, ...)``

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``mca`` (:class:`~torch.Tensor`): A tensor with the accuracy score whose returned shape depends on the
      ``average`` and ``multidim_average`` arguments:

        - If ``multidim_average`` is set to ``global``:

          - If ``average='micro'/'macro'/'weighted'``, the output will be a scalar tensor
          - If ``average=None/'none'``, the shape will be ``(C,)``

        - If ``multidim_average`` is set to ``samplewise``:

          - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(N,)``
          - If ``average=None/'none'``, the shape will be ``(N, C)``

    Args:
        num_classes: Integer specifing the number of classes
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: Calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: Calculates statistic for each label and applies no reduction

        top_k:
            Number of highest probability or logit score predictions considered to find the correct label.
            Only works when ``preds`` contain probabilities/logits.
        multidim_average:
            Defines how additionally dimensions ``...`` should be handled. Should be one of the following:

            - ``global``: Additional dimensions are flatted along the batch dimension
            - ``samplewise``: Statistic will be calculated independently for each sample on the ``N`` axis.
              The statistics in this case are calculated over the additional dimensions.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Example (preds is int tensor):
        >>> from torchmetrics.classification import MulticlassAccuracy
        >>> target = torch.tensor([2, 1, 0, 0])
        >>> preds = torch.tensor([2, 1, 0, 1])
        >>> metric = MulticlassAccuracy(num_classes=3)
        >>> metric(preds, target)
        tensor(0.8333)
        >>> mca = MulticlassAccuracy(num_classes=3, average=None)
        >>> mca(preds, target)
        tensor([0.5000, 1.0000, 1.0000])

    Example (preds is float tensor):
        >>> from torchmetrics.classification import MulticlassAccuracy
        >>> target = torch.tensor([2, 1, 0, 0])
        >>> preds = torch.tensor([
        ...   [0.16, 0.26, 0.58],
        ...   [0.22, 0.61, 0.17],
        ...   [0.71, 0.09, 0.20],
        ...   [0.05, 0.82, 0.13],
        ... ])
        >>> metric = MulticlassAccuracy(num_classes=3)
        >>> metric(preds, target)
        tensor(0.8333)
        >>> mca = MulticlassAccuracy(num_classes=3, average=None)
        >>> mca(preds, target)
        tensor([0.5000, 1.0000, 1.0000])

    Example (multidim tensors):
        >>> from torchmetrics.classification import MulticlassAccuracy
        >>> target = torch.tensor([[[0, 1], [2, 1], [0, 2]], [[1, 1], [2, 0], [1, 2]]])
        >>> preds = torch.tensor([[[0, 2], [2, 0], [0, 1]], [[2, 2], [2, 1], [1, 0]]])
        >>> metric = MulticlassAccuracy(num_classes=3, multidim_average='samplewise')
        >>> metric(preds, target)
        tensor([0.5000, 0.2778])
        >>> mca = MulticlassAccuracy(num_classes=3, multidim_average='samplewise', average=None)
        >>> mca(preds, target)
        tensor([[1.0000, 0.0000, 0.5000],
                [0.0000, 0.3333, 0.5000]])
    """
    is_differentiable = False
    higher_is_better = True
    full_state_update: bool = False

    def compute(self) -> Tensor:
        """Computes accuracy based on inputs passed in to ``update`` previously."""
        tp, fp, tn, fn = self._final_state()
        return _accuracy_reduce(tp, fp, tn, fn, average=self.average, multidim_average=self.multidim_average)

class MulticlassFBetaScore(MulticlassStatScores):
    r"""Computes `F-score`_ metric for multiclass tasks:

    .. math::
        F_{\beta} = (1 + \beta^2) * \frac{\text{precision} * \text{recall}}
        {(\beta^2 * \text{precision}) + \text{recall}}

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, ...)`` or float tensor of shape ``(N, C, ..)``.
      If preds is a floating point we apply ``torch.argmax`` along the ``C`` dimension to automatically convert
      probabilities/logits into an int tensor.
    - ``target`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, ...)``.


    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``mcfbs`` (:class:`~torch.Tensor`): A tensor whose returned shape depends on the ``average`` and
      ``multidim_average`` arguments:

        - If ``multidim_average`` is set to ``global``:

          - If ``average='micro'/'macro'/'weighted'``, the output will be a scalar tensor
          - If ``average=None/'none'``, the shape will be ``(C,)``

        - If ``multidim_average`` is set to ``samplewise``:

          - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(N,)``
          - If ``average=None/'none'``, the shape will be ``(N, C)``

    Args:
        beta: Weighting between precision and recall in calculation. Setting to 1 corresponds to equal weight
        num_classes: Integer specifing the number of classes
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: Calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: Calculates statistic for each label and applies no reduction
        top_k:

            Number of highest probability or logit score predictions considered to find the correct label.
            Only works when ``preds`` contain probabilities/logits.
        multidim_average:
            Defines how additionally dimensions ``...`` should be handled. Should be one of the following:

            - ``global``: Additional dimensions are flatted along the batch dimension
            - ``samplewise``: Statistic will be calculated independently for each sample on the ``N`` axis.
              The statistics in this case are calculated over the additional dimensions.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Example (preds is int tensor):
        >>> from torchmetrics.classification import MulticlassFBetaScore
        >>> target = torch.tensor([2, 1, 0, 0])
        >>> preds = torch.tensor([2, 1, 0, 1])
        >>> metric = MulticlassFBetaScore(beta=2.0, num_classes=3)
        >>> metric(preds, target)
        tensor(0.7963)
        >>> mcfbs = MulticlassFBetaScore(beta=2.0, num_classes=3, average=None)
        >>> mcfbs(preds, target)
        tensor([0.5556, 0.8333, 1.0000])

    Example (preds is float tensor):
        >>> from torchmetrics.classification import MulticlassFBetaScore
        >>> target = torch.tensor([2, 1, 0, 0])
        >>> preds = torch.tensor([
        ...   [0.16, 0.26, 0.58],
        ...   [0.22, 0.61, 0.17],
        ...   [0.71, 0.09, 0.20],
        ...   [0.05, 0.82, 0.13],
        ... ])
        >>> metric = MulticlassFBetaScore(beta=2.0, num_classes=3)
        >>> metric(preds, target)
        tensor(0.7963)
        >>> mcfbs = MulticlassFBetaScore(beta=2.0, num_classes=3, average=None)
        >>> mcfbs(preds, target)
        tensor([0.5556, 0.8333, 1.0000])

    Example (multidim tensors):
        >>> from torchmetrics.classification import MulticlassFBetaScore
        >>> target = torch.tensor([[[0, 1], [2, 1], [0, 2]], [[1, 1], [2, 0], [1, 2]]])
        >>> preds = torch.tensor([[[0, 2], [2, 0], [0, 1]], [[2, 2], [2, 1], [1, 0]]])
        >>> metric = MulticlassFBetaScore(beta=2.0, num_classes=3, multidim_average='samplewise')
        >>> metric(preds, target)
        tensor([0.4697, 0.2706])
        >>> mcfbs = MulticlassFBetaScore(beta=2.0, num_classes=3, multidim_average='samplewise', average=None)
        >>> mcfbs(preds, target)
        tensor([[0.9091, 0.0000, 0.5000],
                [0.0000, 0.3571, 0.4545]])
    """
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False

    def __init__(
        self,
        beta: float,
        num_classes: int,
        top_k: int = 1,
        average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
        multidim_average: Literal["global", "samplewise"] = "global",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            top_k=top_k,
            average=average,
            multidim_average=multidim_average,
            ignore_index=ignore_index,
            validate_args=False,
            **kwargs,
        )
        if validate_args:
            _multiclass_fbeta_score_arg_validation(beta, num_classes, top_k, average, multidim_average, ignore_index)
        self.validate_args = validate_args
        self.beta = beta

    def compute(self) -> Tensor:
        tp, fp, tn, fn = self._final_state()
        return _fbeta_reduce(tp, fp, tn, fn, self.beta, average=self.average, multidim_average=self.multidim_average)

class MulticlassF1Score(MulticlassFBetaScore):
    r"""Computes F-1 score for multiclass tasks:

    .. math::
        F_{1} = 2\frac{\text{precision} * \text{recall}}{(\text{precision}) + \text{recall}}

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, ...)`` or float tensor of shape ``(N, C, ..)``.
      If preds is a floating point we apply ``torch.argmax`` along the ``C`` dimension to automatically convert
      probabilities/logits into an int tensor.
    - ``target`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, ...)``


    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``mcf1s`` (:class:`~torch.Tensor`): A tensor whose returned shape depends on the ``average`` and
      ``multidim_average`` arguments:

        - If ``multidim_average`` is set to ``global``:

          - If ``average='micro'/'macro'/'weighted'``, the output will be a scalar tensor
          - If ``average=None/'none'``, the shape will be ``(C,)``

        - If ``multidim_average`` is set to ``samplewise``:

          - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(N,)``
          - If ``average=None/'none'``, the shape will be ``(N, C)``

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_classes: Integer specifing the number of classes
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: Calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: Calculates statistic for each label and applies no reduction
        top_k:
            Number of highest probability or logit score predictions considered to find the correct label.
            Only works when ``preds`` contain probabilities/logits.
        multidim_average:
            Defines how additionally dimensions ``...`` should be handled. Should be one of the following:

            - ``global``: Additional dimensions are flatted along the batch dimension
            - ``samplewise``: Statistic will be calculated independently for each sample on the ``N`` axis.
              The statistics in this case are calculated over the additional dimensions.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Example (preds is int tensor):
        >>> from torchmetrics.classification import MulticlassF1Score
        >>> target = torch.tensor([2, 1, 0, 0])
        >>> preds = torch.tensor([2, 1, 0, 1])
        >>> metric = MulticlassF1Score(num_classes=3)
        >>> metric(preds, target)
        tensor(0.7778)
        >>> mcf1s = MulticlassF1Score(num_classes=3, average=None)
        >>> mcf1s(preds, target)
        tensor([0.6667, 0.6667, 1.0000])

    Example (preds is float tensor):
        >>> from torchmetrics.classification import MulticlassF1Score
        >>> target = torch.tensor([2, 1, 0, 0])
        >>> preds = torch.tensor([
        ...   [0.16, 0.26, 0.58],
        ...   [0.22, 0.61, 0.17],
        ...   [0.71, 0.09, 0.20],
        ...   [0.05, 0.82, 0.13],
        ... ])
        >>> metric = MulticlassF1Score(num_classes=3)
        >>> metric(preds, target)
        tensor(0.7778)
        >>> mcf1s = MulticlassF1Score(num_classes=3, average=None)
        >>> mcf1s(preds, target)
        tensor([0.6667, 0.6667, 1.0000])

    Example (multidim tensors):
        >>> from torchmetrics.classification import MulticlassF1Score
        >>> target = torch.tensor([[[0, 1], [2, 1], [0, 2]], [[1, 1], [2, 0], [1, 2]]])
        >>> preds = torch.tensor([[[0, 2], [2, 0], [0, 1]], [[2, 2], [2, 1], [1, 0]]])
        >>> metric = MulticlassF1Score(num_classes=3, multidim_average='samplewise')
        >>> metric(preds, target)
        tensor([0.4333, 0.2667])
        >>> mcf1s = MulticlassF1Score(num_classes=3, multidim_average='samplewise', average=None)
        >>> mcf1s(preds, target)
        tensor([[0.8000, 0.0000, 0.5000],
                [0.0000, 0.4000, 0.4000]])
    """
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: int,
        top_k: int = 1,
        average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
        multidim_average: Literal["global", "samplewise"] = "global",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            beta=1.0,
            num_classes=num_classes,
            top_k=top_k,
            average=average,
            multidim_average=multidim_average,
            ignore_index=ignore_index,
            validate_args=validate_args,
            **kwargs,
        )
