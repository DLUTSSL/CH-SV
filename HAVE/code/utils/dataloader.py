import os
import pickle
import h5py
import jieba
import jieba.analyse as analyse
import numpy as np
import pandas as pd
import torch
from scipy.spatial import distance
from sklearn import preprocessing
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset
from transformers import BertTokenizer, AutoModel, AutoTokenizer
        

class CHSVDataset(Dataset):

    def __init__(self, path_vid, datamode='title+ocr'):  # path_vid就是train val test 对应的视频id txt文件

        self.data_complete = pd.read_json(r'/root/autodl-tmp/SVdetection/features/dataset_final.json',orient='records')
        self.framefeapath=r'/root/autodl-tmp/SVdetection/features/frames_fea/'
        self.c3dfeapath=r'/root/autodl-tmp/SVdetection/features/video_fea/'
        self.audiofeapath = r'/root/autodl-tmp/SVdetection/features/audio_fea/'

        self.vid = [] # 初始化一个空列表，用于存放视频ID
        
        with open(r'/root/autodl-tmp/SVdetection/features/vid_shuffle/' + path_vid, "r") as fr:
            for line in fr.readlines():
                self.vid.append(line.strip()) # 读取指定路径下的视频ID文件，并添加到self.vid列表中
        self.data_complete['video_id'] = self.data_complete['video_id'].astype(str)
        self.data = self.data_complete[self.data_complete.video_id.isin(self.vid)]  # 根据读取到的视频ID列表，从完整数据集中筛选出对应的数据
        self.data['video_id'] = self.data['video_id'].astype('category') # 将'video_id'列设置为分类类型，并设置其分类为self.vid中的值
        self.data['video_id'].cat.set_categories(self.vid, inplace=True)
        self.data.sort_values('video_id', ascending=True, inplace=True)    
        self.data.reset_index(inplace=True)  

        self.tokenizer = BertTokenizer.from_pretrained(r'/root/autodl-tmp/SVdetection/bert-base-chinese')

        self.datamode = datamode
        
    def __len__(self):
        print(f"Dataset length: {len(self.data['video_id'])}")
        return self.data.shape[0]
     
    def __getitem__(self, idx):
        item = self.data.iloc[idx] # 按索引idx检索一行数据，并将其存储在变量 item 中。
        vid = item['video_id']

        # label 
        ## label = 0 if item['annotation']=='真' else 1
        label = {
            '正常': 0,
            '暴力': 1,
            '危险': 2,
            '色情': 3,
            '虚假': 4,
            '冒犯': 5
        }.get(item['annotation'], -1)
        label = torch.tensor(label)

        # title+transcript # 根据self.datamode的值决定如何对文本进行预处理
        if self.datamode == 'title+ocr':
            #title_tokens = self.tokenizer(item['title']+' '+item['ocr'], max_length=512, padding='max_length', truncation=True)
            title = item['title'] if item['title'] else ""  # 如果是 None，则替换为空字符串
            ocr = item['ocr'] if item['ocr'] else ""  # 同理处理 OCR
            title_tokens = self.tokenizer(title + " " + ocr, max_length=512, padding='max_length', truncation=True)
        elif self.datamode == 'ocr':
            title_tokens = self.tokenizer(item['ocr'], max_length=512, padding='max_length', truncation=True)
        elif self.datamode == 'title':
            title_tokens = self.tokenizer(item['title'], max_length=512, padding='max_length', truncation=True)
        title_inputid = torch.LongTensor(title_tokens['input_ids']) # 从分词结果中提取 input_ids（即每个词的 ID），并将其转换为 PyTorch 长整型张量。
        title_mask = torch.LongTensor(title_tokens['attention_mask']) # 提取 attention_mask（用于指示哪些位置是实际的输入，哪些位置是填充的）

        # comments
        #comment_tokens = self.tokenizer(item['comments'], max_length=512, padding='max_length', truncation=True)
        # 确保 comments 不为空，否则赋值为空字符串 ""
        comments_text = item['comments'] if item['comments'] else ""
        # 使用 tokenizer 处理文本
        comment_tokens = self.tokenizer(comments_text, max_length=512, padding='max_length', truncation=True)
        comments_inputid = torch.LongTensor(comment_tokens['input_ids'])
        comments_mask = torch.LongTensor(comment_tokens['attention_mask'])
        
        # audio
        audioframes = pickle.load(open(os.path.join(self.audiofeapath,vid+'.pkl'),'rb'))
        audioframes = torch.FloatTensor(audioframes)
        
        # frames
        frames=pickle.load(open(os.path.join(self.framefeapath,vid+'.pkl'),'rb'))
        frames=torch.FloatTensor(frames)
        
        # video
        c3d = pickle.load(open(os.path.join(self.c3dfeapath,vid+'.pkl'),'rb'))
        c3d = torch.FloatTensor(c3d)

        # responses_entity
        response_en_tokens = self.tokenizer(item['主体分析'], max_length=512, padding='max_length', truncation=True)
        response_en_inputid = torch.LongTensor(response_en_tokens['input_ids'])
        response_en_mask = torch.LongTensor(response_en_tokens['attention_mask'])

        # responses_content
        response_con_tokens = self.tokenizer(item['内容分析'], max_length=512, padding='max_length', truncation=True)
        response_con_inputid = torch.LongTensor(response_con_tokens['input_ids'])
        response_con_mask = torch.LongTensor(response_con_tokens['attention_mask'])

        # print("Label:", label.shape)
        # print("Title input ids:", title_inputid.shape)
        # print("Title mask:", title_mask.shape)
        # print("Audio frames:", audioframes.shape)
        # print("Frames:", frames.shape)
        # print("C3D:", c3d.shape)
        # print("Comments input ids:", comments_inputid.shape)
        # print("Comments mask:", comments_mask.shape)
        # print("Comments like:", comments_like.shape)
        # print("Intro input ids:", intro_inputid.shape)
        # print("Intro mask:", intro_mask.shape)

        return {
            'label': label,
            'title_inputid': title_inputid,
            'title_mask': title_mask,
            'audioframes': audioframes,
            'frames':frames,
            'c3d': c3d,
            'comments_inputid': comments_inputid,
            'comments_mask': comments_mask,
            'response_en_inputid': response_en_inputid,
            'response_en_mask': response_en_mask,
            'response_con_inputid': response_con_inputid,
            'response_con_mask': response_con_mask,
        }

