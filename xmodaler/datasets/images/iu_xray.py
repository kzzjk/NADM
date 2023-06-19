"""
@author: Guosheng Zhao
"""
import os
import copy
import pickle
import random
from tqdm import tqdm
import numpy as np
from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.functional import read_np, dict_as_tensor, boxes_to_locfeats
from ..build import DATASETS_REGISTRY
from PIL import Image
import torch
from torchvision import transforms
from sklearn.preprocessing import MultiLabelBinarizer
__all__ = ["IUxrayDataset"]


@DATASETS_REGISTRY.register()
class IUxrayDataset:
    @configurable
    def __init__(
            self,
            stage: str,
            anno_file: str,
            image_file: str,
            seq_per_img: int,
            max_feat_num: int,
            max_seq_len: int,
            feats_folder: str,
            class_nun: int,
            label_file: str,
    ):
        self.stage = stage
        self.anno_file = anno_file
        self.image_file = image_file
        self.seq_per_img = seq_per_img
        self.max_feat_num = max_feat_num
        self.feats_folder = feats_folder
        self.max_seq_len = max_seq_len
        self.class_nun = class_nun
        self.label_file = label_file
        self.mlb = MultiLabelBinarizer(classes=np.arange(class_nun))
        if stage == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                #transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=10, translate=(0.2, 0.2), scale=(0.8, 1.2), fill=(0, 0, 0)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
    @classmethod
    def from_config(cls, cfg, stage: str = "train"):
        ann_files = {
            "train": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "iu_caption_anno_train.pkl"),
            "val": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "iu_caption_anno_val.pkl"),
            "test": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "iu_caption_anno_test.pkl")
        }
        ret = {
            "stage": stage,
            "anno_file": ann_files[stage],
            "class_nun": cfg.MODEL.SEMICNET.NUM_CLASSES,
            "image_file": cfg.DATALOADER.IMAGE_FOLDER,
            "label_file": cfg.DATALOADER.LABEL_FOLDER,
            "seq_per_img": cfg.DATALOADER.SEQ_PER_SAMPLE,
            "max_feat_num": cfg.DATALOADER.MAX_FEAT_NUM,
            "feats_folder": cfg.DATALOADER.FEATS_FOLDER,
            "max_seq_len": cfg.MODEL.MAX_SEQ_LEN
        }
        return ret

    def _preprocess_datalist(self, datalist):
        return datalist

    def load_data(self, cfg):
        def _load_pkl_file(filepath):
            return pickle.load(open(filepath, 'rb'), encoding='bytes') if len(filepath) > 0 else None

        datalist = pickle.load(open(self.anno_file, 'rb'), encoding='bytes')
        datalist = self._preprocess_datalist(datalist)
        return datalist

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image_id = dataset_dict['image_id']
        if len(self.image_file) > 0:
            image_path = dataset_dict['image_path']
            image_1 = Image.open(os.path.join(self.image_file, image_path[0])).convert('RGB')
            image_2 = Image.open(os.path.join(self.image_file, image_path[1])).convert('RGB')
            if self.transform is not None:
                image_1 = self.transform(image_1)
                image_2 = self.transform(image_2)
            image = torch.stack((image_1, image_2), 0)
            ret = {kfg.IDS: image_id, kfg.IMAGE_FEATS: image}

        elif len(self.feats_folder) > 0:
            # feat_path  = os.path.join(self.feats_folder, '100001.npz')
            feat_path0 = os.path.join(self.feats_folder, image_id + '_0.npz')
            feat_path1 = os.path.join(self.feats_folder, image_id + '_1.npz')
            content0 = read_np(feat_path0)
            content1 = read_np(feat_path1)
            att_feats0 = content0['features'][0:self.max_feat_num].astype('float32')
            att_feats1 = content1['features'][0:self.max_feat_num].astype('float32')
            att_feats = np.concatenate((att_feats0, att_feats1), axis=0)
            ret = {kfg.IDS: image_id, kfg.ATT_FEATS: att_feats}


        if self.stage != 'train':
            g_tokens_type = np.ones((self.max_seq_len,), dtype=np.int64)
            ret.update({kfg.G_TOKENS_TYPE: g_tokens_type,kfg.STAGE: self.stage})
            dict_as_tensor(ret)
            return ret

        sent_num = len(dataset_dict['tokens_ids'])
        if sent_num >= self.seq_per_img:
            selects = random.sample(range(sent_num), self.seq_per_img)
        else:
            selects = random.choices(range(sent_num), k=(self.seq_per_img - sent_num))
            selects += list(range(sent_num))

        tokens_ids = [dataset_dict['tokens_ids'][i, :].astype(np.int64) for i in selects]
        target_ids = [dataset_dict['target_ids'][i, :].astype(np.int64) for i in selects]
        g_tokens_type = [np.ones((len(dataset_dict['tokens_ids'][i, :]),), dtype=np.int64) for i in selects]

        abnormal_label = [dataset_dict['abnormal_label'].astype(np.int64)]
        normal_label = [dataset_dict['normal_label'].astype(np.int64)]

        abnormal_label = self.mlb.fit_transform(abnormal_label)
        normal_label = self.mlb.fit_transform(normal_label)

        ret.update({
            kfg.STAGE: self.stage,
            kfg.SEQ_PER_SAMPLE: self.seq_per_img,
            kfg.G_TOKENS_IDS: tokens_ids,
            kfg.G_TARGET_IDS: target_ids,
            kfg.G_TOKENS_TYPE: g_tokens_type,
            kfg.NORMAL_LABEL: normal_label,
            kfg.ABNORMAL_LABEL: abnormal_label,

        })

        dict_as_tensor(ret)
        return ret

