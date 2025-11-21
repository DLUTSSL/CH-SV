import collections
import json
import os
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from gensim.models import KeyedVectors

from models.HaveModel import *
from utils.dataloader import *
# from models.Trainer import Trainer
# from models.Trainer2 import Trainer3

from transformers import AutoModel, AutoTokenizer
from have import *
def pad_sequence(seq_len, lst, emb):
    result = []
    for video in lst:
        if isinstance(video, list):
            video = torch.stack(video)
        ori_len = video.shape[0]
        if ori_len == 0:
            video = torch.zeros([seq_len, emb], dtype=torch.long)
        elif ori_len >= seq_len:
            if emb == 200:
                video = torch.FloatTensor(video[:seq_len])
            else:
                video = torch.LongTensor(video[:seq_len])
        else:
            video = torch.cat([video, torch.zeros([seq_len - ori_len, video.shape[1]], dtype=torch.long)], dim=0)
            if emb == 200:
                video = torch.FloatTensor(video)
            else:
                video = torch.LongTensor(video)
        result.append(video)
    return torch.stack(result)

def pad_frame_sequence(seq_len, lst):
    attention_masks = []
    result = []
    first_video_features = None

    for video in lst:
        if len(video) > 0 and first_video_features is None:
            # 将第一个非空视频的特征赋值给 first_video_features
            first_video_features = torch.FloatTensor(video)

        if len(video) == 0:  # 如果视频长度为0
            if first_video_features is not None:
                # 如果有非空视频，则使用第一个视频的特征形状作为模板来创建全零张量
                video = torch.zeros((seq_len, first_video_features.shape[1]), dtype=torch.float)
            else:
                # 如果没有非空视频，则创建默认的全零张量
                video = torch.zeros((seq_len, 1), dtype=torch.float)
            mask = torch.zeros(seq_len, dtype=torch.int)  # 创建全零的注意力掩码
        else:
            video = torch.FloatTensor(video)  # 将numpy数组转换为PyTorch张量
            ori_len = video.shape[0]  # 获取原始视频的长度
            if ori_len >= seq_len:
                gap = ori_len // seq_len
                video = video[::gap][:seq_len]
                mask = torch.ones(seq_len, dtype=torch.int)  # 创建全1的注意力掩码
            else:
                # 使用第一个视频的特征形状作为模板来创建全零张量
                video = torch.cat(
                    (video, torch.zeros([seq_len - ori_len, first_video_features.shape[1]], dtype=torch.float)), dim=0)
                mask = torch.cat(
                    (torch.ones(ori_len, dtype=torch.int), torch.zeros(seq_len - ori_len, dtype=torch.int)), dim=0)
        result.append(video)
        attention_masks.append(mask)

    # 统一所有视频的形状
    max_feature_size = max([v.shape[1] for v in result])
    for i in range(len(result)):
        diff = max_feature_size - result[i].shape[1]
        if diff > 0:
            padding = torch.zeros((seq_len, diff), dtype=torch.float)
            result[i] = torch.cat((result[i], padding), dim=1)

    return torch.stack(result), torch.stack(attention_masks)


def _init_fn(worker_id):
    np.random.seed(2022)


def CHSV_collate_fn(batch):
    num_frames = 83
    num_audioframes = 50

    response_en_inputid = [item['response_en_inputid'] for item in batch]
    response_en_mask = [item['response_en_mask'] for item in batch]

    response_con_inputid = [item['response_con_inputid'] for item in batch]
    response_con_mask = [item['response_con_mask'] for item in batch]

    title_inputid = [item['title_inputid'] for item in batch]
    title_mask = [item['title_mask'] for item in batch]

    comments_inputid = [item['comments_inputid'] for item in batch]
    comments_mask = [item['comments_mask'] for item in batch]

    frames = [item['frames'] for item in batch]
    frames, frames_masks = pad_frame_sequence(num_frames, frames)

    audioframes = [item['audioframes'] for item in batch]
    audioframes, audioframes_masks = pad_frame_sequence(num_audioframes, audioframes)

    c3d = [item['c3d'] for item in batch]
    c3d, c3d_masks = pad_frame_sequence(num_frames, c3d)

    label = [item['label'] for item in batch]
    # print('After collect_fn: ')
    # print("Label:", torch.stack(label).shape)
    # print("Response_entity Input ID:", torch.stack(response_en_inputid).shape)
    # print("Response_entity Mask:", torch.stack(response_en_mask).shape)
    # print("Response_content Input ID:", torch.stack(response_con_inputid).shape)
    # print("Response_content Mask:", torch.stack(response_con_mask).shape)
    # print("Title Input ID:", torch.stack(title_inputid).shape)
    # print("Title Mask:", torch.stack(title_mask).shape)
    # print("Comment Input ID:", torch.stack(comments_inputid).shape)
    # print("Comment Mask:", torch.stack(comments_mask).shape)
    # print("Audio Frames:", audioframes.shape)
    # print("Audio Frames Masks:", audioframes_masks.shape)
    # print("Frames:", frames.shape)
    # print("Frames Masks:", frames_masks.shape)
    # print("C3D:", c3d.shape)
    # print("C3D Masks:", c3d_masks.shape)


    return {
        'label': torch.stack(label),
        'response_en_inputid': torch.stack(response_en_inputid),
        'response_en_mask': torch.stack(response_en_mask),
        'response_con_inputid': torch.stack(response_con_inputid),
        'response_con_mask': torch.stack(response_con_mask),
        'title_inputid': torch.stack(title_inputid),
        'title_mask': torch.stack(title_mask),
        'comments_inputid': torch.stack(comments_inputid),
        'comments_mask': torch.stack(comments_mask),
        'audioframes': audioframes,
        'audioframes_masks': audioframes_masks,
        'frames': frames,
        'frames_masks': frames_masks,
        'c3d': c3d,
        'c3d_masks': c3d_masks,
    }



class Run():
    def __init__(self,
                 config
                 ):

        self.model_name = config['model_name']
        self.mode_eval = config['mode_eval']
        self.fold = config['fold']
        self.data_type = 'HAVE'

        self.epoches = config['epoches']
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.epoch_stop = config['epoch_stop']
        self.seed = config['seed']
        self.device = config['device']
        self.lr = config['lr']
        self.lambd = config['lambd']
        self.save_param_dir = config['path_param']
        self.path_tensorboard = config['path_tensorboard']
        self.dropout = config['dropout']
        self.weight_decay = config['weight_decay']
        self.event_num = 616
        self.mode = 'normal'


    def get_dataloader_temporal(self, data_type):
        collate_fn = None
        if data_type == 'HAVE':
            dataset_train = CHSVDataset('vid_train.txt')
            dataset_val = CHSVDataset('vid_val.txt')
            dataset_test = CHSVDataset('vid_test.txt')
            collate_fn = CHSV_collate_fn

        train_dataloader = DataLoader(dataset_train, batch_size=self.batch_size,
                                      num_workers=self.num_workers,
                                      pin_memory=True,
                                      shuffle=True,
                                      worker_init_fn=_init_fn,
                                      collate_fn=collate_fn)
        val_dataloader = DataLoader(dataset_val, batch_size=self.batch_size,
                                    num_workers=self.num_workers,
                                    pin_memory=True,
                                    shuffle=False,
                                    worker_init_fn=_init_fn,
                                    collate_fn=collate_fn)
        test_dataloader = DataLoader(dataset_test, batch_size=self.batch_size,
                                     num_workers=self.num_workers,
                                     pin_memory=True,
                                     shuffle=False,
                                     worker_init_fn=_init_fn,
                                     collate_fn=collate_fn)

        dataloaders = dict(zip(['train', 'val', 'test'], [train_dataloader, val_dataloader, test_dataloader]))

        return dataloaders


    def get_have_model(self):
        self.model = HaveModel(bert_model=r'/root/autodl-tmp/SVdetection/bert-base-chinese', fea_dim=128, dropout=0.3)
        return self.model

    def main(self):
        if self.mode_eval == "temporal":
            have_model = self.get_have_model()
            dataloaders = self.get_dataloader_temporal(data_type=self.data_type)
            trainer = Trainer3(model=have_model, device=self.device, lr=self.lr, dataloaders=dataloaders,
                                            epoches=self.epoches, dropout=self.dropout, weight_decay=self.weight_decay,
                                            mode=self.mode, model_name=self.model_name, event_num=self.event_num,
                                            epoch_stop=self.epoch_stop,
                                            save_param_path=self.save_param_dir + self.data_type + "/" + self.model_name + "/",
                                            writer=SummaryWriter(self.path_tensorboard))
            result = trainer.train()
            return result
