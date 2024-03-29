import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
import spacy

torch.set_printoptions(threshold=float('inf'))
# def min_max_norm(data_dict):
#     minm = 1000.
#     maxm = .0
#     for key in data_dict:
#         if data_dict[key] < minm:
#             minm = data_dict[key]
#         if data_dict[key] > maxm:
#             maxm = data_dict[key]
#     con = maxm - minm
#     for key in data_dict:
#         data_dict[key] = (data_dict[key] - minm) / con
#     return data_dict

        
# def cal_tf_idf(documents, tokenizer):
#     sents_num = len(documents)
#     dic_tf = {}
#     dic_idf = {}
#     tf_idf = {}
#     for i in range(sents_num):
#         count = 0
#         sents = documents[i].split()
#         dic = {}
#         dic_tmp_tf = {}
#         for j in sents:
#             x = tokenizer.get_id_by_token(j)
#             if x not in dic_tmp_tf:
#                 dic_tmp_tf[x] = 1
#             else:
#                 dic_tmp_tf[x] += 1
        
#             dic[x] = 1
#             count += 1
#         for j in dic_tmp_tf:
#             # print(j)
#             if j not in dic_tf:
#                 # dic_tf[j] = [np.log(1 + dic_tmp_tf[j]/count)] ### log
#                 dic_tf[j] = [dic_tmp_tf[j]/count]
#             else:
#                 # dic_tf[j].append(np.log(1 + dic_tmp_tf[j]/count))
#                 dic_tf[j].append(dic_tmp_tf[j]/count)
        
             
#         for j in dic:
#             if j not in dic_idf:
#                 dic_idf[j] = 1
#             else:
#                 dic_idf[j] += 1
                
#     for i in dic_idf:
#         dic_idf[i] = np.log(sents_num / dic_idf[i])
#     # print(dic_tf)
#     for i in dic_tf:
#         s = 0
#         for j in dic_tf[i]:
#             s += j
#         dic_tf[i] = s / len(dic_tf[i])
#     # print(dic_tf)
#     for i in dic_idf:
#         tf_idf[i] = dic_idf[i] * dic_tf[i]
#     # print(dic_idf)
#     # print()
#     # print(dic_tf)
#     # print()
#     # print(tf_idf[1])
#     tf_idf = min_max_norm(tf_idf)
#     # print()
#     # print(tf_idf)
#     for i in tf_idf:
#         if tf_idf[i] < 0.1:
#             print(tokenizer.get_token_by_id(i))
#     exit()
#     # print(dic_tf)
#     # draw_plot(dic_tf)
#     # exit()
#     result = []
#     for i in range(sents_num):
#         result.append([])
#         sents = documents[i].split()
#         for j in sents:
#             result[-1].append(dic_tf[j] * dic_idf[j])

def pos_tagging(documents, tokenizer):
    nlp = spacy.load("en_core_web_sm")  # 加载英文模型
    pos_dict = {}  # 存储单词和词性的字典
    # pos_word_dict = {}  # 存储词性和对应单词列表的字典
    
    for doc in documents:
        doc = nlp(doc)  # 对文档进行处理

        for token in doc:
            word = token.text
            pos = token.pos_

            pos_dict[tokenizer.get_id_by_token(word)] = pos
            

            # if pos in pos_word_dict:
            #     if word not in pos_word_dict[pos]:
            #         pos_word_dict[pos].append(word)
            # else:
            #     pos_word_dict[pos] = [word]
    # with open('./data/iu_xray/pos_dict.json', 'w') as f:
        # json.dump(pos_dict, f)
    # print(pos_dict)
    # exit()
    # return pos_dict



class BaseDataset(Dataset):
    def __init__(self, path, split):
        with open(path, 'r') as f:
            data = json.load(f)
            self.samples = data[split]
        self.samples['input'] = torch.tensor(np.array(self.samples['input']))
        self.samples['label'] = torch.tensor(np.array(self.samples['label']))
        self.data_filter()
        # print(self.samples['input'].size())
        print(self.samples['input'].any())
        print(self.samples['label'])
        exit()
    def data_filter(self):
        
        each_stock_input = self.samples['input'].transpose(0, 2)
        each_stock_input = each_stock_input.reshape(each_stock_input.size(0), 430 * 30)
        each_stock_label = self.samples['label'].transpose(0, 1)
        filt = torch.ones(each_stock_input.size(0), dtype=torch.bool)
        for i in range(each_stock_input.size(0)):
            # print(each_stock_input[0])
            zero_num = torch.eq(each_stock_input[0], 0)
            # print(zero_num)
            zero_num = torch.sum(zero_num)
            print(zero_num)
            x = input()
            if x == '1':
                pass
            else:
                exit()
            if zero_num > (each_stock_input.size(0) * 0.1):
                filt[i] = 0
            else:
                print(each_stock_input[i])
                
        each_stock_input = each_stock_input[filt, :]
        each_stock_label = each_stock_label[filt, :]
        print('input size:',each_stock_input.size())
        print('label size:',each_stock_label.size())
        exit()

    def __len__(self):
        return len(self.samples['input'])

    def __getitem__(self, idx):
        input_data = self.samples['input'][idx]
        gtr = self.samples['label'][idx]
        
        return (input_data, gtr)
data_file = '../data.json'
split = 'train'
dataset = BaseDataset(data_file, split)


# class IuxrayMultiImageDataset(BaseDataset):
#     def __getitem__(self, idx):
#         example = self.examples[idx]
#         image_id = example['id']
#         image_path = example['image_path']
        

#         image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
#         image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
#         if self.transform is not None:
#             image_1 = self.transform(image_1) # [3, 224, 224]
#             image_2 = self.transform(image_2)

#         image = torch.stack((image_1, image_2), 0) # [2图片, 3, 224, 224]
#         report_ids = example['ids']
#         report_masks = example['mask']
#         seq_length = len(report_ids) 
#         # sample = (image_id, image, report_ids, report_masks, seq_length, tf_idf)
#         sample = (image_id, image, report_ids, report_masks, seq_length)
#         return sample


# class MimiccxrSingleImageDataset(BaseDataset):
#     def __getitem__(self, idx):
#         example = self.examples[idx]
#         image_id = example['id']
#         image_path = example['image_path']

#         image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
#         image_id = os.path.join(self.image_dir, image_path[0])
#         if self.transform is not None:
#             image = self.transform(image) # [3, 224, 224]

#         report_ids = example['ids']
#         report_masks = example['mask']
#         seq_length = len(report_ids)
#         sample = (image_id, image, report_ids, report_masks, seq_length)
#         return sample