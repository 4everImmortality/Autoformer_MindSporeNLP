import os
import mindspore.context as context
from mindspore import Tensor

class Exp_Basic:
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError("Subclasses should implement this method.")
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
            print('Use GPU')
        else:
            context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
            print('Use CPU')
        return "GPU" if self.args.use_gpu else "CPU"

    def _get_data(self):
        # This method can be implemented to load and preprocess data.
        pass

    def vali(self):
        # This method can be implemented to perform validation.
        pass

    def train(self):
        # This method can be implemented to perform training.
        pass

    def test(self):
        # This method can be implemented to perform testing.
        pass

# import os
# import torch


# class Exp_Basic(object):
#     def __init__(self, args):
#         self.args = args
#         self.device = self._acquire_device()
#         self.model = self._build_model().to(self.device)

#     def _build_model(self):
#         raise NotImplementedError
#         return None

#     def _acquire_device(self):
#         if self.args.use_gpu:
#             os.environ["CUDA_VISIBLE_DEVICES"] = str(
#                 self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
#             device = torch.device('cuda:{}'.format(self.args.gpu))
#             print('Use GPU: cuda:{}'.format(self.args.gpu))
#         else:
#             device = torch.device('cpu')
#             print('Use CPU')
#         return device

#     def _get_data(self):
#         pass

#     def vali(self):
#         pass

#     def train(self):
#         pass

#     def test(self):
#         pass
