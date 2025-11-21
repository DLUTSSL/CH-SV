import os
import pandas as pd
import numpy as np
import argparse
import random
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

from sklearn.metrics import precision_recall_fscore_support

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from transformers import get_linear_schedule_with_warmup
from transformers import RobertaTokenizer, RobertaModel
import gc

# from preprocessing import *
# from utils import *
# from dataset import *
from models.HaveModel import *
from utils.metrics import *
from data_t import *


def _SaveModel(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    model_state_dict = model.state_dict() #修改过的
    torch.save(model_state_dict, os.path.join(path, 'have.bin'))


class Trainer3():
    def __init__(self, model, device, lr, dropout, dataloaders, weight_decay, save_param_path, writer, epoch_stop, epoches, mode,
                 model_name, event_num, save_threshold=0.0, start_epoch=0,):

        self.model = model
        self.device = device
        self.mode = mode
        self.model_name = model_name
        self.event_num = event_num

        self.dataloaders = dataloaders
        self.start_epoch = start_epoch
        self.num_epochs = epoches
        self.epoch_stop = epoch_stop
        self.save_threshold = save_threshold
        self.writer = writer

        if os.path.exists(save_param_path):
            self.save_param_path = save_param_path
        else:
            self.save_param_path = os.makedirs(save_param_path)
            self.save_param_path = save_param_path

        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        since = time.time()
        self.model.cuda()

        best_model_wts_val = copy.deepcopy(self.model.state_dict())
        best_acc_val = 0.0
        best_epoch_val = 0

        is_earlystop = False

        for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):
            if is_earlystop:
                break
            print('-' * 50)
            print('Epoch {}/{}'.format(epoch + 1, self.start_epoch + self.num_epochs))
            print('-' * 50)

            p = float(epoch) / 100
            lr = self.lr / (1. + 10 * p) ** 0.75
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr, weight_decay=self.weight_decay)

            for phase in ['train', 'val', 'test']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                print('-' * 10)
                print(phase.upper())
                print('-' * 10)

                running_loss = 0.0
                tpred = []
                tlabel = []


                for batch in tqdm(self.dataloaders[phase]):
                    batch_data = batch
                    for k, v in batch_data.items():
                        batch_data[k] = v.cuda()
                    label = batch_data['label']
                    if self.mode == "eann":
                        label_event = batch_data['label_event']

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        fea, outputs = self.model(batch_data['audioframes'], batch_data['audioframes_masks'],
                                                batch_data['frames'], batch_data['frames_masks'], batch_data['c3d'],
                                                batch_data['c3d_masks'], batch_data['title_inputid'], batch_data['title_mask'],                                                           batch_data['comments_inputid'], batch_data['comments_mask'],                                                                             batch_data['response_en_inputid'], 
                                                batch_data['response_en_mask'],
                                                batch_data['response_con_inputid'],
                                                batch_data['response_con_mask'])
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, label)

                        if phase == 'train':
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            self.optimizer.step()
                            self.optimizer.zero_grad()

                    tlabel.extend(label.detach().cpu().numpy().tolist())
                    tpred.extend(preds.detach().cpu().numpy().tolist())
                    running_loss += loss.item() * label.size(0)

                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                #print(f"Epoch {epoch + 1} - Phase: {phase}")
                #print(f"真实标签: {tlabel}")
                #print(f"预测标签: {tpred}")
                print('Loss: {:.4f} '.format(epoch_loss))
                results = metrics(tlabel, tpred)
                print(results)

                if phase == 'val':
                    if results['acc'] > best_acc_val:
                        best_acc_val = results['acc']
                        best_model_wts_val = copy.deepcopy(self.model.state_dict())
                        best_epoch_val = epoch + 1
                        if best_acc_val >= self.save_threshold:  # 用最新一轮的指标来保存，即使acc相等的情况下
                            _SaveModel(self.model, self.save_param_path)
                            print("saved " + self.save_param_path + "_val_epoch" + str(best_epoch_val) + "_{0:.4f}".format(
                                best_acc_val))
                    else:
                        if epoch - best_epoch_val >= self.epoch_stop - 1:
                            is_earlystop = True
                            print("early stopping...")

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print("Best model on val: epoch" + str(best_epoch_val) + "_" + str(best_acc_val))

        self.model.load_state_dict(best_model_wts_val)
        print("test result when using best model on val")
        return self.test()

    def test(self):
        since = time.time()

        self.model.cuda()
        self.model.eval()

        pred = []
        label = []

        for batch in tqdm(self.dataloaders['test']):
            with torch.no_grad():
                batch_data = batch
                for k, v in batch_data.items():
                    batch_data[k] = v.cuda()
                batch_label = batch_data['label']

                batch_outputs, fea = self.model(batch_data['audioframes'], batch_data['audioframes_masks'],
                                                batch_data['frames'], batch_data['frames_masks'], batch_data['c3d'],
                                                batch_data['c3d_masks'], batch_data['title_inputid'], batch_data['title_mask'],                                                           batch_data['comments_inputid'], batch_data['comments_mask'],                                                                             batch_data['response_en_inputid'], 
                                                batch_data['response_en_mask'],
                                                batch_data['response_con_inputid'],
                                                batch_data['response_con_mask'])
                _, batch_preds = torch.max(fea, 1)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_preds.detach().cpu().numpy().tolist())

        print(metrics(label, pred))

        return metrics(label, pred)

        #return pred, label


# def evaluation(model, dataloader):
#     model.eval()
#
#     pred = []
#     label = []
#
#     for batch in tqdm(dataloaders['test']):
#         with torch.no_grad():
#             batch_data = batch
#             for k, v in batch_data.items():
#                 batch_data[k] = v.cuda()
#             batch_label = batch_data['label']
#
#             batch_outputs, class_output = model(batch_data['title_inputid'], batch_data['title_mask'],
#                                                   batch_data['intro_inputid'], batch_data['intro_mask'],
#                                                   batch_data['comments_inputid'], batch_data['comments_mask'],
#                                                   batch_data['comments_like'])
#
#             _, batch_preds = torch.max(class_output, 1)
#
#             label.extend(batch_label.detach().cpu().numpy().tolist())
#             pred.extend(batch_preds.detach().cpu().numpy().tolist())
#
#     print(get_confusionmatrix_fnd(np.array(pred), np.array(label)))
#     print(metrics(label, pred))
#     # return metrics(label, pred)
#     return pred, label
