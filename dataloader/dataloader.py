"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

from collections import Counter
import os
import pandas as pd
import torch.utils.data as util_data
from torch.utils.data import Dataset

class TextClustering(Dataset):
    def __init__(self, train_x, train_y):
        assert len(train_x) == len(train_y)
        self.train_x = train_x
        self.train_y = train_y

    def __len__(self):
        return len(self.train_x)

    def __getitem__(self, idx):
        return {'text': self.train_x[idx], 'label': self.train_y[idx]}

class AugmentPairSamples(Dataset):
    def __init__(self, train_x, train_x1, train_x2, train_y):
        assert len(train_y) == len(train_x) == len(train_x1) == len(train_x2)
        self.train_x = train_x
        self.train_x1 = train_x1
        self.train_x2 = train_x2
        self.train_y = train_y
        
    def __len__(self):
        return len(self.train_y)

    def __getitem__(self, idx):
        return {'text': self.train_x[idx], 'text1': self.train_x1[idx], 'text2': self.train_x2[idx], 'label': self.train_y[idx]}


class AugmentPairSamplesWithInfo(Dataset):
    def __init__(self, train_x, train_x1, train_x2, train_y, info):
        assert len(train_y) == len(train_x) == len(train_x1) == len(train_x2)
        self.train_x = train_x
        self.train_x1 = train_x1
        self.train_x2 = train_x2
        self.train_y = train_y
        self.info = info
        
    def __len__(self):
        return len(self.train_y)

    def __getitem__(self, idx):
        return {'text': self.train_x[idx], 'text1': self.train_x1[idx], 'text2': self.train_x2[idx], 'label': self.train_y[idx], 'info': self.info[idx]}


def augment_loader(args, file_format=None, text_column_name=None):
    if file_format == "csv":
        train_data = pd.read_csv(os.path.join(args.data_path, args.dataname), sep=',', error_bad_lines=False)
    else:
        train_data = pd.read_csv(os.path.join(args.data_path, args.dataname), sep='\t', error_bad_lines=False)

    train_text = train_data[text_column_name].fillna('.').values # 原始文本
    text_list = []
    for text in train_text:
        if 'VirtualIslander_MemberJoin' in text or 'GroupChat_CallBack' in text \
            or 'GroupChat_MemberJoin' in text or 'AvaterDoubleClicked' in text \
            or 'AvaterDoubleClicked' in text or 'AvaterDoubleClicked' in text:
            continue
        while ('@' in text and '\u2005' in text):
            index1 = text.find('@')
            index2 = text.find('\u2005')
            while(index1 > index2 and index2 > 0):
                index2_next = text[index2+1:].find("\u2005")
                if index2_next < 0:
                    break
                index2 = index2 + 1 + index2_next
            if index2 < 0 or index2 < index1:
                break
            text = text[:index1] + text[index2+1:]

        text = text.replace('\n','').replace('\t','').replace('\r','').strip()
        at_index = text.find('@')
        if at_index > -1 and len(text) - at_index < 4:
            text = text[:at_index]
            print(f'Filtered: text {text[at_index+1:]} from raw {text}')
        if '@' in text and len(text) < 5:
            print(f'Filtered: text {text} from raw {text}')
        if not text and len(text) < 3:
            continue
        text_list.append(text)
    print(f'sample num:{len(text_list)}')
    train_text = text_list
    train_text1 = text_list    # 增强文本1
    train_text2 = text_list    # 增强文本2
    try:
        train_label = train_data['label'].astype(int).values    # 数据标签
    except:
        train_label = [0] * len(text_list)   # 数据标签

    train_dataset = AugmentPairSamples(train_text, train_text1, train_text2, train_label)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    return train_loader


# def augment_loader_from_log(args, file_format=None, text_column_name=None):
#     if file_format is "csv":
#         train_data = pd.read_csv(os.path.join(args.data_path, args.dataname), sep=',', error_bad_lines=False)
#     else:
#         train_data = pd.read_csv(os.path.join(args.data_path, args.dataname), sep='\t', error_bad_lines=False)

#     train_text = train_data[text_column_name].fillna('.').values # 原始文本
#     train_text1 = train_data[text_column_name].fillna('.').values    # 增强文本1
#     train_text2 = train_data[text_column_name].fillna('.').values    # 增强文本2
#     # try:
#     #     train_label = train_data['label'].astype(int).values    # 数据标签
#     # except:
#     #     train_label = train_data['label'].fillna(0).values    # 数据标签

#     train_dataset = AugmentPairSamples(train_text, train_text1, train_text2, train_label)
#     train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
#     return train_loader



def augment_loader_intention(args):
    fr = open(os.path.join(args.data_path, args.dataname), "r")
    lines = fr.readlines()
    top_train_loader = {}
    train_loader_all = {}
    key_pre = None
    total_cnt = 0
    key_list = []
    for line in lines:
        line_arr = line.strip('\n').split('\t')
        # key = ":".join(line_arr[0:2])
        key = line_arr[1]
        if total_cnt == 0:
            key_pre = key
            train_text = []
            train_label = []
            info = []
        if key_pre != key:
            # 开始新key先将原来的key的数据集收集起来
            train_dataset = AugmentPairSamplesWithInfo(train_text, train_text, train_text, train_label, info)
            train_loader_all[key_pre] = train_dataset 
            train_text = []
            train_label = []
            info = []
            key_pre = key
        key_list.append(key)
        next_request = line_arr[2]
        train_text.append(next_request) # 原始文本
        train_label.append(0)
        info.append("\t".join(line_arr[3:]) if len(line_arr) > 3 else None)
        total_cnt += 1

    key_counter = Counter(key_list)
    key_counter_sort = sorted(key_counter.items(), key=lambda x:x[1], reverse=True)
    topCntKey = 60
    for key_count_pair in key_counter_sort:
        if topCntKey < 0:
            break
        else:
            topCntKey -= 1
        key = key_count_pair[0]
        cnt = key_count_pair[1]
        if key not in train_loader_all:
            print('Key Error '+key)
            continue
        train_dataset = train_loader_all[key]
        train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
        top_train_loader[f'{key}_{cnt}'] = train_loader
    
    return top_train_loader


def augment_loader_onlin_log(args):
    train_data = pd.read_csv(os.path.join(args.data_path, args.dataname))
    text = train_data['RequestContent'].str.replace('\n',' ').replace('\t',' ').fillna('.').values
    train_text = text # 原始文本
    train_text1 = text    # 增强文本1
    train_text2 = text    # 增强文本2
    train_label = [0] * len(train_text)    # 数据标签

    train_dataset = AugmentPairSamples(train_text, train_text1, train_text2, train_label)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    return train_loader


def train_unshuffle_loader(args):
    train_data = pd.read_csv(os.path.join(args.data_path, args.dataname), sep='\t')
    train_text = train_data['text'].fillna('.').values
    train_label = train_data['label'].astype(int).values

    train_dataset = TextClustering(train_text, train_label)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)   
    return train_loader

