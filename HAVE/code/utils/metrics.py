from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score, recall_score, precision_score  
from sklearn.preprocessing import label_binarize  
from sklearn.metrics import roc_curve, auc  
from itertools import combinations  
import numpy as np

def metrics(y_true, y_pred):  
    # print("y_true:", y_true)  
    # print("y_pred:", y_pred)
    metrics = {}  
  
    # 准确度  
    metrics['acc'] = accuracy_score(y_true, y_pred)  
  
    # 召回率、精确度和F1分数（macro平均）  
    metrics['f1'] = f1_score(y_true, y_pred, average='macro')  
    metrics['recall'] = recall_score(y_true, y_pred, average='macro')  
    metrics['precision'] = precision_score(y_true, y_pred, average='macro')
    
    # 计算各个类别的 F1 分数
    metrics['f1-normal'] = f1_score(y_true, y_pred, average=None, labels=[0])
    metrics['f1-violence'] = f1_score(y_true, y_pred, average=None, labels=[1])
    metrics['f1-danger'] = f1_score(y_true, y_pred, average=None, labels=[2])
    metrics['f1-vulgarity'] = f1_score(y_true, y_pred, average=None, labels=[3])
    metrics['f1-fake'] = f1_score(y_true, y_pred, average=None, labels=[4])
    metrics['f1-offensive'] = f1_score(y_true, y_pred, average=None, labels=[5])
    
    return metrics  
  