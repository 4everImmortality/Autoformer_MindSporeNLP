import mindspore as ms
import mindspore.numpy as mnp
from mindspore import Tensor

class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = (B, 1, L, L)
        self._mask = Tensor(mnp.triu(mnp.ones(mask_shape, dtype=mnp.bool_), k=1))

    @property
    def mask(self):
        return self._mask
    
class ProbMask:
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = Tensor(mnp.ones((L, scores.shape[-1]), dtype=mnp.bool_)).triu(1)
        _mask_ex = _mask[None, None, :].expand((B, H, L, scores.shape[-1]))
        indicator = _mask_ex[
            Tensor(list(range(B)), ms.int32)[:, None, None],
            Tensor(list(range(H)), ms.int32)[None, :, None],
            index,
            :
        ].asnumpy()
        self._mask = Tensor(indicator.reshape(scores.shape))

    @property
    def mask(self):
        return self._mask

# import torch


# class TriangularCausalMask():
#     def __init__(self, B, L, device="cpu"):
#         mask_shape = [B, 1, L, L]
#         with torch.no_grad():
#             self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

#     @property
#     def mask(self):
#         return self._mask


# class ProbMask():
#     def __init__(self, B, H, L, index, scores, device="cpu"):
#         _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
#         _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
#         indicator = _mask_ex[torch.arange(B)[:, None, None],
#                     torch.arange(H)[None, :, None],
#                     index, :].to(device)
#         self._mask = indicator.view(scores.shape).to(device)

#     @property
#     def mask(self):
#         return self._mask
