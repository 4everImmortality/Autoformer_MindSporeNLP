import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)

# MindSpore imports
import mindspore as ms
from mindspore import nn, ops, context, Tensor
from mindspore.train import Model
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore.common.initializer import Normal

import numpy as np
import os
import time

import warnings
warnings.filterwarnings('ignore')

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, Reformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.model = self._build_model()
        self.optimizer = self._select_optimizer()
    
    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'Reformer': Reformer,
        }
        model = model_dict[self.args.model](self.args)

        return model
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        optimizer = nn.Adam(self.model.trainable_params(), learning_rate=self.args.learning_rate)
        return optimizer

    def _select_criterion(self):
        criterion = ms.nn.MSELoss()
        return criterion


    def _predict(self, batch_x, batch_x_mark, batch_y, batch_y_mark):
        # decoder input
        dec_inp = ms.numpy.zeros_like(batch_y[:, -self.args.pred_len:, :]).astype(ms.float32)
        dec_inp = ms.ops.cat([batch_y[:, :self.args.label_len, :], dec_inp], axis=1).astype(ms.float32)
        
        # encoder - decoder
        
        def _run_model():
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if self.args.output_attention:
                outputs = outputs[0]
            return outputs

        if self.args.use_amp:
            with ms.amp.auto_mixed_precision():
                outputs = _run_model()
        else:
            outputs = _run_model()

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

        return outputs, batch_y

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.set_train(False)  # Set the model to evaluation mode
        with ms.ms_function():  # Use ms_function for graph mode execution
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = ms.Tensor(batch_x, dtype=ms.float32)
                batch_y = ms.Tensor(batch_y, dtype=ms.float32)

                batch_x_mark = ms.Tensor(batch_x_mark, dtype=ms.float32)
                batch_y_mark = ms.Tensor(batch_y_mark, dtype=ms.float32)

                outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

                pred = outputs.asnumpy()
                true = batch_y.asnumpy()

                loss = criterion(Tensor(pred), Tensor(true))

                total_loss.append(loss.asnumpy())
        self.model.set_train(True)  # Restore the model to training mode
        total_loss = np.mean(total_loss)
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.set_train(True)  # Set the model to training mode
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                
                batch_x = ms.Tensor(batch_x, dtype=ms.float32)
                batch_y = ms.Tensor(batch_y, dtype=ms.float32)
                batch_x_mark = ms.Tensor(batch_x_mark, dtype=ms.float32)
                batch_y_mark = ms.Tensor(batch_y_mark, dtype=ms.float32)

                outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.asnumpy())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.asnumpy()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.mean(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.ckpt'
        self.model.load_param_dict(ms.load_checkpoint(best_model_path))

        return

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            model_path = os.path.join('./checkpoints/' + setting, 'checkpoint.ckpt')
            self.model.load_param_dict(ms.load_checkpoint(model_path))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.set_train(False)  # Set the model to evaluation mode
        with ms.ms_function():  # Use ms_function for graph mode execution
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = ms.Tensor(batch_x, dtype=ms.float32)
                batch_y = ms.Tensor(batch_y, dtype=ms.float32)

                batch_x_mark = ms.Tensor(batch_x_mark, dtype=ms.float32)
                batch_y_mark = ms.Tensor(batch_y_mark, dtype=ms.float32)

                outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

                outputs = outputs.asnumpy()
                batch_y = batch_y.asnumpy()

                pred = outputs  # .squeeze()
                true = batch_y  # .squeeze()

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.asnumpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        with open("result.txt", 'a') as f:
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}'.format(mse, mae))
            f.write('\n')
            f.write('\n')

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.ckpt'
            logging.info(best_model_path)
            self.model.load_param_dict(ms.load_checkpoint(best_model_path))

        preds = []

        self.model.set_train(False)  # Set the model to evaluation mode
        with ms.ms_function():  # Use ms_function for graph mode execution
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = ms.Tensor(batch_x, dtype=ms.float32)
                batch_y = ms.Tensor(batch_y, dtype=ms.float32)
                batch_x_mark = ms.Tensor(batch_x_mark, dtype=ms.float32)
                batch_y_mark = ms.Tensor(batch_y_mark, dtype=ms.float32)

                outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

                pred = outputs.asnumpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
    
# import logging
# logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
#     datefmt='%Y-%m-%d:%H:%M:%S',
#     level=logging.INFO)

# from data_provider.data_factory import data_provider
# from exp.exp_basic import Exp_Basic
# from models import Informer, Autoformer, Transformer, Reformer
# from utils.tools import EarlyStopping, adjust_learning_rate, visual
# from utils.metrics import metric

# import numpy as np
# import torch
# import torch.nn as nn
# from torch import optim

# import os
# import time

# import warnings
# import numpy as np

# warnings.filterwarnings('ignore')


# class Exp_Main(Exp_Basic):
#     def __init__(self, args):
#         super(Exp_Main, self).__init__(args)

#     def _build_model(self):
#         model_dict = {
#             'Autoformer': Autoformer,
#             'Transformer': Transformer,
#             'Informer': Informer,
#             'Reformer': Reformer,
#         }
#         model = model_dict[self.args.model].Model(self.args).float()

#         if self.args.use_multi_gpu and self.args.use_gpu:
#             model = nn.DataParallel(model, device_ids=self.args.device_ids)
#         return model

#     def _get_data(self, flag):
#         data_set, data_loader = data_provider(self.args, flag)
#         return data_set, data_loader

#     def _select_optimizer(self):
#         model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
#         return model_optim

    # def _select_criterion(self):
    #     criterion = nn.MSELoss()
    #     return criterion

    # def _predict(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
    #     # decoder input
    #     dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
    #     dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
    #     # encoder - decoder

    #     def _run_model():
    #         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    #         if self.args.output_attention:
    #             outputs = outputs[0]
    #         return outputs

    #     if self.args.use_amp:
    #         with torch.cuda.amp.autocast():
    #             outputs = _run_model()
    #     else:
    #         outputs = _run_model()

    #     f_dim = -1 if self.args.features == 'MS' else 0
    #     outputs = outputs[:, -self.args.pred_len:, f_dim:]
    #     batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

    #     return outputs, batch_y

    # def vali(self, vali_data, vali_loader, criterion):
    #     total_loss = []
    #     self.model.eval()
    #     with torch.no_grad():
    #         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
    #             batch_x = batch_x.float().to(self.device)
    #             batch_y = batch_y.float()

    #             batch_x_mark = batch_x_mark.float().to(self.device)
    #             batch_y_mark = batch_y_mark.float().to(self.device)

    #             outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

    #             pred = outputs.detach().cpu()
    #             true = batch_y.detach().cpu()

    #             loss = criterion(pred, true)

    #             total_loss.append(loss)
    #     total_loss = np.average(total_loss)
    #     self.model.train()
    #     return total_loss

    # def train(self, setting):
    #     train_data, train_loader = self._get_data(flag='train')
    #     vali_data, vali_loader = self._get_data(flag='val')
    #     test_data, test_loader = self._get_data(flag='test')

    #     path = os.path.join(self.args.checkpoints, setting)
    #     if not os.path.exists(path):
    #         os.makedirs(path)

    #     time_now = time.time()

    #     train_steps = len(train_loader)
    #     early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

    #     model_optim = self._select_optimizer()
    #     criterion = self._select_criterion()

    #     if self.args.use_amp:
    #         scaler = torch.cuda.amp.GradScaler()

    #     for epoch in range(self.args.train_epochs):
    #         iter_count = 0
    #         train_loss = []

    #         self.model.train()
    #         epoch_time = time.time()
    #         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
    #             iter_count += 1
    #             model_optim.zero_grad()
    #             batch_x = batch_x.float().to(self.device)

    #             batch_y = batch_y.float().to(self.device)
    #             batch_x_mark = batch_x_mark.float().to(self.device)
    #             batch_y_mark = batch_y_mark.float().to(self.device)

    #             outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

    #             loss = criterion(outputs, batch_y)
    #             train_loss.append(loss.item())

    #             if (i + 1) % 100 == 0:
    #                 print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
    #                 speed = (time.time() - time_now) / iter_count
    #                 left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
    #                 print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
    #                 iter_count = 0
    #                 time_now = time.time()

    #             if self.args.use_amp:
    #                 scaler.scale(loss).backward()
    #                 scaler.step(model_optim)
    #                 scaler.update()
    #             else:
    #                 loss.backward()
    #                 model_optim.step()

    #         print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
    #         train_loss = np.average(train_loss)
    #         vali_loss = self.vali(vali_data, vali_loader, criterion)
    #         test_loss = self.vali(test_data, test_loader, criterion)

    #         print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
    #             epoch + 1, train_steps, train_loss, vali_loss, test_loss))
    #         early_stopping(vali_loss, self.model, path)
    #         if early_stopping.early_stop:
    #             print("Early stopping")
    #             break

    #         adjust_learning_rate(model_optim, epoch + 1, self.args)

    #     best_model_path = path + '/' + 'checkpoint.pth'
    #     self.model.load_state_dict(torch.load(best_model_path))

    #     return

    # def test(self, setting, test=0):
    #     test_data, test_loader = self._get_data(flag='test')
    #     if test:
    #         print('loading model')
    #         self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

    #     preds = []
    #     trues = []
    #     folder_path = './test_results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)

    #     self.model.eval()
    #     with torch.no_grad():
    #         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
    #             batch_x = batch_x.float().to(self.device)
    #             batch_y = batch_y.float().to(self.device)

    #             batch_x_mark = batch_x_mark.float().to(self.device)
    #             batch_y_mark = batch_y_mark.float().to(self.device)

    #             outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

    #             outputs = outputs.detach().cpu().numpy()
    #             batch_y = batch_y.detach().cpu().numpy()

    #             pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
    #             true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

    #             preds.append(pred)
    #             trues.append(true)
    #             if i % 20 == 0:
    #                 input = batch_x.detach().cpu().numpy()
    #                 gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
    #                 pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
    #                 visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

    #     preds = np.concatenate(preds, axis=0)
    #     trues = np.concatenate(trues, axis=0)
    #     print('test shape:', preds.shape, trues.shape)
    #     preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    #     trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    #     print('test shape:', preds.shape, trues.shape)

    #     # result save
    #     folder_path = './results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)

    #     mae, mse, rmse, mape, mspe = metric(preds, trues)
    #     print('mse:{}, mae:{}'.format(mse, mae))
    #     f = open("result.txt", 'a')
    #     f.write(setting + "  \n")
    #     f.write('mse:{}, mae:{}'.format(mse, mae))
    #     f.write('\n')
    #     f.write('\n')
    #     f.close()

    #     np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
    #     np.save(folder_path + 'pred.npy', preds)
    #     np.save(folder_path + 'true.npy', trues)

    #     return

    # def predict(self, setting, load=False):
    #     pred_data, pred_loader = self._get_data(flag='pred')

    #     if load:
    #         path = os.path.join(self.args.checkpoints, setting)
    #         best_model_path = path + '/' + 'checkpoint.pth'
    #         logging.info(best_model_path)
    #         self.model.load_state_dict(torch.load(best_model_path))

    #     preds = []

    #     self.model.eval()
    #     with torch.no_grad():
    #         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
    #             batch_x = batch_x.float().to(self.device)
    #             batch_y = batch_y.float()
    #             batch_x_mark = batch_x_mark.float().to(self.device)
    #             batch_y_mark = batch_y_mark.float().to(self.device)

    #             outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

    #             pred = outputs.detach().cpu().numpy()  # .squeeze()
    #             preds.append(pred)

    #     preds = np.array(preds)
    #     preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

    #     # result save
    #     folder_path = './results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)

    #     np.save(folder_path + 'real_prediction.npy', preds)

    #     return
