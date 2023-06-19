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
normal_class=['normal','normally','clear', 'sharp', 'sharply', 'unremarkable', 'intact', 'stable', 'free','unchanged','inflate']
ignore_obs=['xxxx','within','focal','size','2','this','acute','change','expand',
            'mild','finding','none','area','contrast','loss','significant','interval','density',
            'limit','marking','overlie','large','flatten','great','small','elevation','grossly','prominent',
            'prominence','elevate','engorgement','otherwise','similar','new','newly','more']
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
def semic_de(dff,semic_threshold):
    df_obs = list(dff['obs_lemma'])
    df_ana = list(dff['anatomy'])
    df_label = list(dff['label'])
    obs_abnormal_list=[]
    obs_normal_list=[]
    ana_abnormal_list=[]
    ana_normal_list=[]
    obs_abnormal_ignore=[]
    obs_normal_ignore=[]
    ana_abnormal_ignore=[]
    ana_normal_ignore=[]
    for obs_i, anatomy_i, label_i in zip(df_obs, df_ana, df_label):
        anatomy_i = anatomy_i.split('|')
        if obs_i not in normal_class and obs_i != 'none' and label_i == 'OBS-DP' and obs_i not in ignore_obs:
            obs_abnormal_list.append(obs_i)
            for anatomy_ii in anatomy_i:
                ana_abnormal_list.append(anatomy_ii)
        elif obs_i not in ignore_obs and obs_i != 'none':
            obs_normal_list.append(obs_i)
            for anatomy_ii in anatomy_i:
                ana_normal_list.append(anatomy_ii)
    result = Counter(obs_abnormal_list)
    sorted_abnormal_obs = sorted(result.items(), key=lambda x: x[1], reverse=True)
    result = Counter(obs_normal_list)
    sorted_normal_obs = sorted(result.items(), key=lambda x: x[1], reverse=True)
    result = Counter(ana_abnormal_list)
    sorted_abnormal_ana = sorted(result.items(), key=lambda x: x[1], reverse=True)
    result = Counter(ana_normal_list)
    sorted_normal_ana = sorted(result.items(), key=lambda x: x[1], reverse=True)
    for data in sorted_abnormal_obs:
        count_num = data[1]
        if count_num<semic_threshold:
            obs_abnormal_ignore.append(data)
    for data in sorted_normal_obs:
        count_num = data[1]
        if count_num < semic_threshold:
            obs_normal_ignore.append(data)
    for data in sorted_abnormal_ana:
        count_num = data[1]
        if count_num < semic_threshold:
            ana_abnormal_ignore.append(data)
    for data in sorted_normal_ana:
        count_num = data[1]
        if count_num < semic_threshold:
            ana_normal_ignore.append(data)
    return obs_abnormal_ignore,obs_normal_ignore,ana_abnormal_ignore,ana_normal_ignore



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
    semic_threshold = args.semic_threshold
    obs_abnormal_ignore,obs_normal_ignore,ana_abnormal_ignore,ana_normal_ignore=semic_de(df_anatomy_label,semic_threshold)

    ann_xray = json.load(open(args.annotation, 'r'))

    label_list= {}
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
            report_abnormal=[]
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
                for obs_i, anatomy_i, label_i, observation_i in zip(obs_lemma, anatomy, label_lemma,observation):
                    anatomy_i=anatomy_i.split('|')
                    if obs_i not in normal_class and obs_i !='none' and label_i =='OBS-DP' and obs_i not in ignore_obs:
                        if obs_i not in obs_abnormal_ignore:
                            if obs_i in vocab:
                                abnormal_word = vocab.index(obs_i)
                                report_abnormal.append(abnormal_word)
                            elif observation_i in vocab:
                                abnormal_word = vocab.index(observation_i)
                                report_abnormal.append(abnormal_word)
                        for anatomy_ii in anatomy_i:
                            if anatomy_ii not in ana_abnormal_ignore:
                                if anatomy_ii in vocab:
                                    abnormal_word = vocab.index(anatomy_ii)
                                    report_abnormal.append(abnormal_word)
                    elif obs_i not in ignore_obs and obs_i !='none':
                        for anatomy_ii in anatomy_i:
                            if anatomy_ii not in ana_normal_ignore:
                                if anatomy_ii in vocab:
                                    normal_word = vocab.index(anatomy_ii)
                                    report_normal.append(normal_word)
                        if obs_i not in obs_normal_ignore:
                            if obs_i not in ignore_obs:
                                if obs_i in vocab:
                                    normal_word = vocab.index(obs_i)
                                    report_normal.append(normal_word)
                                elif observation_i in vocab:
                                    normal_word = vocab.index(observation_i)
                                    report_normal.append(normal_word)
            if len(report_normal) ==0:
                report_normal.append(0)
            if len(report_abnormal) ==0:
                report_abnormal.append(0)
            label_data = {
                "normal_label": report_normal,
                "abnormal_label": report_abnormal,
            }
            label_list[name] = label_data
    pkl.dump(label_list, open(args.save_path, "wb"))
def add_semic(args):
    datalist_train = pickle.load(open(args.data_train, 'rb'), encoding='bytes')
    datalist_val = pickle.load(open(args.data_val, 'rb'), encoding='bytes')
    datalist_test = pickle.load(open(args.data_test, 'rb'), encoding='bytes')
    semic = pickle.load(open(args.semic, 'rb'), encoding='bytes')
    datalist = {"train": [], "val": [], "test": []}
    datalist['train'] = datalist_train
    datalist['val'] = datalist_val
    datalist['test'] = datalist_test
    for k,datalistk in datalist.items():
        for data in datalistk:
            if args.name == 'mimic-cxr':
                image_id = data['study_id']
            else:
                image_id = data['image_id']
            semic_normal = semic[image_id]['normal_label']
            semic_abnormal = semic[image_id]['abnormal_label']
            data['normal_label'] = np.array(semic_normal)
            data['abnormal_label'] = np.array(semic_abnormal)
        pkl.dump(datalistk, open(args.semic_save+ k + ".pkl", "wb"))

def main(args):
    seed(123)
    vocab = load_vocab(args.vocab_path)
    print('vocab:',len(vocab))
    semic_get(args,vocab)
    add_semic(args)
    print('finish')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str,
                        default='mimic-cxr',
                        help='path for train annotation file')
    parser.add_argument('--annotation', type=str,
                        default='/home/zgs/X-ray/mimic_cxr/new_data/mimic_ann_plus.json',
                        help='path for train annotation file')
    parser.add_argument('--save_path', type=str, default='/home/zgs/X-ray/mimic_cxr/new_data/mimic_semiclabel.pkl',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--vocab_path', type=str, default='/home/zgs/X-ray/mimic_cxr/new_data/mimic_vocabulary.txt',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--radgraph', type=str, default='/home/zgs/X-ray/mimic_cxr/new_data/mimic-cxr-radgraph-sentence-parsed.csv',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--data_train', type=str, default='/home/zgs/X-ray/mimic_cxr/new_data/mimic_caption_anno_train.pkl',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--data_val', type=str, default='/home/zgs/X-ray/mimic_cxr/new_data/mimic_caption_anno_val.pkl',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--data_test', type=str, default='/home/zgs/X-ray/mimic_cxr/new_data/mimic_caption_anno_test.pkl',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--semic', type=str,
                        default='/home/zgs/X-ray/mimic_cxr/new_data/mimic_semiclabel.pkl',
                        help='path for train annotation file')
    parser.add_argument('--semic_save', type=str,
                        default='/home/zgs/X-ray/mimic_cxr/new_data/mimic_caption_anno_',
                        help='path for train annotation file')
    parser.add_argument('--semic_threshold', type=int,
                        default=30,
                        help='path for train annotation file')

    # parser.add_argument('--name', type=str,
    #                     default='iu_xray',
    #                     help='path for train annotation file')
    # parser.add_argument('--annotation', type=str,
    #                     default='/home/zgs/X-ray/iu_xray/annotation.json',
    #                     help='path for train annotation file')
    # parser.add_argument('--save_path', type=str, default='/home/zgs/X-ray/iu_xray/new_data/iu_semiclabel.pkl',
    #                     help='path for saving vocabulary wrapper')
    # parser.add_argument('--vocab_path', type=str, default='/home/zgs/X-ray/iu_xray/new_data/iu_vocabulary.txt',
    #                     help='path for saving vocabulary wrapper')
    # parser.add_argument('--radgraph', type=str, default='/home/zgs/X-ray/iu_xray/new_data/iu-radgraph-sentence-parsed.csv',
    #                     help='path for saving vocabulary wrapper')
    # parser.add_argument('--data_train', type=str, default='/home/zgs/X-ray/iu_xray/new_data/iu_caption_anno_train.pkl',
    #                     help='path for saving vocabulary wrapper')
    # parser.add_argument('--data_val', type=str, default='/home/zgs/X-ray/iu_xray/new_data/iu_caption_anno_val.pkl',
    #                     help='path for saving vocabulary wrapper')
    # parser.add_argument('--data_test', type=str, default='/home/zgs/X-ray/iu_xray/new_data/iu_caption_anno_test.pkl',
    #                     help='path for saving vocabulary wrapper')
    # parser.add_argument('--semic', type=str,
    #                     default='/home/zgs/X-ray/iu_xray/new_data/iu_semiclabel.pkl',
    #                     help='path for train annotation file')
    # parser.add_argument('--semic_save', type=str,
    #                     default='/home/zgs/X-ray/iu_xray/new_data/iu_caption_anno_',
    #                     help='path for train annotation file')
    # parser.add_argument('--semic_threshold', type=int,
    #                     default=10,
    #                     help='path for train annotation file')

    args = parser.parse_args()
    main(args)

