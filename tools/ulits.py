import json
import torch
import numpy as np
import re
import pickle
import pandas as pd
import nltk
import pickle
import argparse
from collections import Counter
import json
import numpy as np
from tqdm import tqdm
import re
import os
from random import shuffle,seed
import pickle as pkl


normal_class=['normal', 'clear', 'sharp', 'sharply', 'unremarkable', 'intact', 'stable', 'free','unchanged','inflate']
ignore_obs=['xxxx','within','focal','size','2','this','acute','change','expand',
            'mild','finding','none','area','contrast','loss','significant','interval','density',
            'limit','marking','overlie','large','flatten','great','small','elevation','grossly','prominent',
            'prominence','elevate','engorgement']
obs_class=['pneumothorax','effusion','consolidation','opacity','degenerative','infiltrate','granuloma','atelectasis','low',
           'nodule','edema','calcified','scar','fracture','enlarge','cardiomegaly','pneumonia','tortuous','air','atherosclerotic',
           'catheter','sternotomy','adenopathy','spondylosis','emphysema','scoliosis','hernia','masses','blunt']
def load_vocab(path):
    if len(path) == 0:
        return None
    vocab = []
    with open(path, 'r') as fid:
        for line in fid:
            vocab.append(line.strip())
    return vocab
def semic_get(args,vocab):
    df_anatomy_label = pd.read_csv(args.radgraph)
    idx_replace = df_anatomy_label['obs_lemma'].isin(['enlargement', 'increase'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'enlarge'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['engorge'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'engorgement'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['opacification', 'opacity-'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'opacity'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['calcify'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'calcification'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['effusion ;'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'effusion'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['atelectatic', 'atelectasis ;', 'atelectase'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'atelectasis'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['aeration'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'aerate'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['distend', 'distension'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'distention'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['wide'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'widen'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['prominent'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'prominence'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['haze'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'haziness'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['masse'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'mass'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['kyphotic'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'kyphosis'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['degenerate'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'degenerative'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['obscuration'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'obscure'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['fibrotic'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'fibrosis'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['nodular', 'nodularity'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'nodule'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['ventilate'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'ventilation'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['tortuosity'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'tortuous'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['elongate'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'elongation'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['elevate'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'elevation'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['drain'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'drainage'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['deviate'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'deviation'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['consolidative', 'consolidate'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'consolidation'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['dilate', 'dilatation'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'dilation'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['hydropneumothorax', 'pneumothoraces', 'pneumothorace'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'pneumothorax'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['improvement', 'improved'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'improve'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['can not be assess', 'can not be evaluate', 'not well see',
                                                      'not well assess', 'can not be accurately assess',
                                                      'not well evaluate', 'not well visualize',
                                                      'difficult to evaluate',
                                                      'poorly see'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'difficult to assess'

    idx_replace = df_anatomy_label['obs_lemma'] == 'pacer'
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'pacemaker'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['infection', 'infectious', 'infectious process'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'pneumonia'

    df_anatomy_label.loc[df_anatomy_label['label'].isna(), 'label'] = 'OBS-NA'
    ann_xray = json.load(open(args.annotation, 'r'))
    all_normal=[]
    all_abnormal = []

    semic_normal= {}
    semic_abnormal = {}
    for k, v in ann_xray.items():
        split = k
        examples = ann_xray[split]
        for i in range(len(examples)):
            if args.name == 'mimic-cxr':
                name = examples[i]['study_id']
            else:
                name = examples[i]['id']
            idx_a = (df_anatomy_label['study_id'] == name)
            df = df_anatomy_label[idx_a]
            if len(list(df['sen_id']))==0:
                print(name)
                continue
            sent_start = list(df['sen_id'])[0]
            sent_len = list(df['sen_id'])[-1]+1
            report_abnormal = []
            report_normal = []
            for sen_i in range(int(sent_start),int(sent_len)):
                idx_b = (df['sen_id'] == sen_i)
                df_b = df[idx_b]
                anatomy = list(df_b['anatomy'])
                label_lemma = list(df_b['label'])
                obs_lemma = list(df_b['obs_lemma'])
                observation =  list(df_b['observation'])
                if len(list(df_b['sentence']))==0:
                    continue
                report_sent = list(df_b['sentence'])[0]
                sent_abnormal = []
                sent_normal = []
                for obs_i, anatomy_i, label_i, observation_i in zip(obs_lemma, anatomy, label_lemma,observation):
                    anatomy_i=anatomy_i.split('|')
                    if obs_i not in normal_class and obs_i !='none' and label_i =='OBS-DP' and obs_i not in ignore_obs:
                        if obs_i in vocab:
                            sent_abnormal.append(obs_i)
                        elif observation_i in vocab:
                            sent_abnormal.append(observation_i)
                        for anatomy_ii in anatomy_i:
                            if anatomy_ii in vocab:
                                sent_abnormal.append(anatomy_ii)
                    else:
                        for anatomy_ii in anatomy_i:
                            if anatomy_ii in vocab:
                                sent_normal.append(anatomy_ii)
                        if obs_i not in ignore_obs:
                            if obs_i in vocab:
                                sent_normal.append(obs_i)
                            elif observation_i in vocab:
                                sent_normal.append(observation_i)
                if len(sent_normal)>0:
                    key = ' '.join(sent_normal)
                    semic_normal[key] = report_sent
                if len(sent_abnormal)>0:
                    key = ' '.join(sent_abnormal)
                    semic_abnormal[key] = report_sent
                report_normal.append(sent_normal)
                report_abnormal.append(sent_abnormal)
        all_normal.append(report_normal)
        all_abnormal.append(report_abnormal)
    json.dump(semic_normal, open('./semic_normal.json', 'w'))
    json.dump(semic_abnormal, open('./semic_abnormal.json', 'w'))
    # result = Counter(report_normal)
    # sorted_normal = sorted(result.items(), key=lambda x: x[1], reverse=True)
    # result = Counter(report_abnormal)
    # sorted_abnormal = sorted(result.items(), key=lambda x: x[1], reverse=True)
    print(1)

def main(args):
    vocab = load_vocab(args.vocab_path)
    print('vocab:',len(vocab))
    semic_get(args,vocab)
    print('finish')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--name', type=str,
    #                     default='mimic-cxr',
    #                     help='path for train annotation file')
    # parser.add_argument('--annotation', type=str,
    #                     default='/home/zgs/X-ray/mimic_cxr/new_data/mimic_ann_plus.json',
    #                     help='path for train annotation file')
    # parser.add_argument('--save_path', type=str, default='/home/zgs/X-ray/mimic_cxr/new_data/mimic_semiclabel.pkl',
    #                     help='path for saving vocabulary wrapper')
    # parser.add_argument('--vocab_path', type=str, default='/home/zgs/X-ray/mimic_cxr/new_data/mimic_vocabulary.txt',
    #                     help='path for saving vocabulary wrapper')
    # parser.add_argument('--radgraph', type=str, default='/home/zgs/X-ray/mimic_cxr/new_data/mimic-cxr-radgraph-sentence-parsed.csv',
    #                     help='path for saving vocabulary wrapper')

    parser.add_argument('--name', type=str,
                        default='iu_xray',
                        help='path for train annotation file')
    parser.add_argument('--annotation', type=str,
                        default='/home/zgs/X-ray/iu_xray/annotation.json',
                        help='path for train annotation file')
    parser.add_argument('--save_path', type=str, default='/home/zgs/X-ray/iu_xray/new_data/iu_semiclabel.pkl',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--vocab_path', type=str, default='/home/zgs/X-ray/iu_xray/new_data/iu_vocabulary.txt',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--radgraph', type=str, default='/home/zgs/X-ray/iu_xray/new_data/iu-radgraph-sentence-parsed.csv',
                        help='path for saving vocabulary wrapper')

    args = parser.parse_args()
    main(args)

