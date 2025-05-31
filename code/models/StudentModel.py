import copy
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import tqdm
from sklearn.metrics import *
from tqdm import tqdm
from transformers import AutoConfig, BertModel
from transformers.models.bert.modeling_bert import BertLayer
from zmq import device
from transformers import BertTokenizer, AutoModel, AutoTokenizer
from utils.metrics import *
from src.CrossmodalTransformer import MULTModel
from src.StoG_different_modal import CapsuleSequenceToGraph

class AudioVideoTeacherModel(nn.Module):
    def __init__(self, bert_model, fea_dim, dropout):
        super(AudioVideoTeacherModel, self).__init__()

        self.video_dim = 4096
        self.img_dim = 4096
        # self.audio_dim = 12288
        self.num_frames = 83
        self.num_audioframes = 50
        self.dim = fea_dim
        self.num_heads = 4
        self.l_in = 128
        self.a_in = 128
        self.v_in = 128
        self.f_in = 128
        self.dropout = dropout
        self.T_t = 512
        self.T_a = 50
        self.T_v = 83
        self.T_f = 83
        self.vertex_num = 42 #32
        self.routing = 4
        self.bert = BertModel.from_pretrained(bert_model).requires_grad_(False)
        self.classifier = nn.Linear(128, 6)

        # Define layers for audio modality
        self.vggish_layer = torch.hub.load(r'/root/autodl-tmp/SVdetection/data/harritaylor_torchvggish_master', 'vggish', source = 'local')
        net_structure = list(self.vggish_layer.children())      
        self.vggish_modified = nn.Sequential(*net_structure[-2:-1])
        
        self.linear_audio = nn.Sequential(
            nn.Linear(128, fea_dim),
            nn.ReLU(),
            nn.Dropout(p=0.25))

        self.linear_video = nn.Sequential(
            nn.Linear(self.video_dim, fea_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2))

        self.linear_text = nn.Sequential(
            nn.Linear(768, fea_dim),
            nn.ReLU(),
            nn.Dropout(p=0.35))

        self.linear_res_con = nn.Sequential(
            nn.Linear(768, fea_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2))

        self.linear_res_en = nn.Sequential(
            nn.Linear(768, fea_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2))

        self.linear_comment = nn.Sequential(
            nn.Linear(768, fea_dim),
            nn.ReLU(),
            nn.Dropout(p=0.35))

        self.linear_img = nn.Sequential(torch.nn.Linear(self.img_dim, fea_dim), torch.nn.ReLU(),nn.Dropout(p=0.2))

        self.CrossmodalTransformer = MULTModel(self.l_in, self.a_in, self.v_in, self.f_in, fea_dim) # MULT
        self.fc_t = nn.Linear(fea_dim * 3, fea_dim)
        self.fc_a = nn.Linear(fea_dim * 3, fea_dim)
        self.fc_v = nn.Linear(fea_dim * 3, fea_dim)
        self.fc_f = nn.Linear(fea_dim * 3, fea_dim)
        self.StoG = CapsuleSequenceToGraph(fea_dim, fea_dim, self.vertex_num, self.routing, self.T_t, self.T_a, self.T_v, self.T_f)
        self.trm = nn.TransformerEncoderLayer(d_model=fea_dim, nhead=2, batch_first=True)

    def forward(self, audioframes, audioframes_masks, frames, frames_masks, c3d, c3d_masks, title_inputid, title_mask, comments_inputid, comments_mask, response_en_inputid, response_en_mask, response_con_inputid, response_con_mask):
        # 处理音频 [batch, 50, 128]
        fea_audio = self.vggish_modified(audioframes)
        fea_audio = self.linear_audio(fea_audio)
        fea_audio = fea_audio.to(torch.float32)

        # 处理视频 [batch, 83, 128]
        fea_video = self.linear_video(c3d)

        # 处理帧 [batch, 83, 128]
        fea_img = self.linear_img(frames)

        # 处理title+ocr [batch, 512, 128]
        fea_title = self.bert(title_inputid, attention_mask=title_mask)['last_hidden_state']  # (batch,sequence,768)
        fea_title = self.linear_text(fea_title)

        # 处理主题分析response [batch, 512, 128]
        fea_response_en = self.bert(response_en_inputid, attention_mask=response_en_mask)['last_hidden_state']  # (batch,sequence,768)
        fea_response_en = self.linear_res_en(fea_response_en)

        # 处理主题分析response [batch, 512, 128]
        fea_response_con = self.bert(response_con_inputid, attention_mask=response_con_mask)['last_hidden_state']  # (batch,sequence,768)
        fea_response_con = self.linear_res_con(fea_response_con)

        # 处理comment [batch, 128]
        #fea_comment = self.bert(comments_inputid, attention_mask=comments_mask)[1]  # (batch,768)
        fea_comment = self.bert(comments_inputid, attention_mask=comments_mask)['last_hidden_state']
        fea_comment = self.linear_comment(fea_comment)

        # fea_response = torch.cat((fea_response_en, fea_response_con), 2) #  [batch, 512, 256]
        # fea_response = self.fc(fea_response)  # 形状为[batch_size, 512, fea_dim]

        z_t, z_a, z_v, z_f = self.CrossmodalTransformer(fea_title, fea_audio, fea_video, fea_img)  # [seq_len,batch_size,3*fea_dim]

        z_t = z_t.transpose(1, 0)
        z_a = z_a.transpose(1, 0)
        z_v = z_v.transpose(1, 0)
        z_f = z_f.transpose(1, 0)  # 形状为[batch_size,seq_len,3*fea_dim] 
        
        z_t = self.fc_t(z_t)
        z_a = self.fc_a(z_a)
        z_v = self.fc_v(z_v)
        z_f = self.fc_f(z_f)  # 形状为[batch_size,seq_len,fea_dim]
        
        x_t, x_a, x_v, x_f = self.StoG(z_t, z_a, z_v, z_f)  # 形状为[batch_size,32,fea_dim]
        
        x_t = torch.mean(x_t, -2)
        x_a = torch.mean(x_a, -2)
        x_v = torch.mean(x_v, -2)
        x_f = torch.mean(x_f, -2)  # 形状为[batch_size,fea_dim]

        x_t = x_t.unsqueeze(1)
        x_a = x_a.unsqueeze(1)
        x_v = x_v.unsqueeze(1)
        x_f = x_f.unsqueeze(1)  # 形状为[batch_size, 1, fea_dim]

        fea_comment = torch.mean(fea_comment, -2)
        fea_comment = fea_comment.unsqueeze(1)  # 形状为[batch_size, 1, fea_dim]
        fea_response_en = torch.mean(fea_response_en, -2)  
        fea_response_con = torch.mean(fea_response_con, -2)  
        fea_response_en = fea_response_en.unsqueeze(1)  # 形状为[batch_size, 1, fea_dim]
        fea_response_con = fea_response_con.unsqueeze(1)  # 形状为[batch_size, 1, fea_dim]
    
        # fea_t = torch.cat((fea_comment, fea_response_en, fea_response_con), 2)  # 形状为[batch_size, 1, 3*fea_dim]

        fea = torch.cat((x_t, x_a, x_v, x_f, fea_comment, fea_response_en, fea_response_con), 1) # 形状为[batch_size, 7, fea_dim]
        
        fea = self.trm(fea)  # 形状为[batch_size, 7, fea_dim]
        fea = torch.mean(fea, -2) # 形状为[batch_size, fea_dim]
        output = self.classifier(fea) # 形状为[batch_size, 6]

        return fea, output



