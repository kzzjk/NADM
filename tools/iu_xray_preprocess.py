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

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def clean_report_iu_xray(report):
    report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
        .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
        .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
        .strip().lower().split('. ')
    sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                    replace('\\', '').replace("'", '').strip().lower())
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
    report = ' . '.join(tokens) + ' .'
    return report


def build_vocab(cap_path, threshold):
    """Build a simple vocabulary wrapper."""
    captions = json.load(open(cap_path, 'r'))

    counter = Counter()
    for k, v in captions.items():
        split = k
        examples = captions[split]
        for i in range(len(examples)):
            name = examples[i]['id']
            report = clean_report_iu_xray(examples[i]['report'])
            tokens = nltk.tokenize.word_tokenize(report.lower())
            counter.update(tokens)

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    vocab = {'idx': vocab.idx, 'word2idx': vocab.word2idx, 'idx2word': vocab.idx2word}
    return vocab


def cap2idx(args, vocab):
    data = json.load(open(args.annotation, 'r'))
    train_gt = {}
    datalist = {"train": [], "val": [], "test": []}
    max_len = 60

    for k, v in data.items():
        split = k
        examples = data[split]
        for i in range(len(examples)):
            idx = examples[i]['id']
            image_path = examples[i]['image_path']
            report = clean_report_iu_xray(examples[i]['report'])
            tokens = nltk.tokenize.word_tokenize(report.lower())
            word2idx = [vocab['word2idx'][ii] if ii in vocab['word2idx'] else vocab['word2idx']['<unk>'] for ii in tokens]
            input = [0] + word2idx
            if len(word2idx) > max_len - 1:
                target = word2idx[:max_len - 1] + [0]
            else:
                target = word2idx + [0]
            if len(input) < max_len:
                while len(input) < max_len:
                    input.append(0)
            else:
                input = input[:max_len]

            if len(target) < max_len:
                while len(target) < max_len:
                    target.append(-1)
            else:
                target = target[:max_len]

            if len(word2idx) < max_len:
                target_gt = word2idx + [0]
            else:
                target_gt = word2idx[:max_len]
            new_data = {
                "image_id": idx,
                "image_path": image_path,
                "tokens_ids": np.array([input]),
                "target_ids": np.array([target])
            }
            datalist[split].append(new_data)
            if split=='train':
                train_gt[idx] = [target_gt]
    pkl.dump(train_gt, open(os.path.join(args.save_path, "iu_train_gts.pkl"), "wb"))
    for split in datalist:
        pkl.dump(datalist[split], open(os.path.join(args.save_path, "iu_caption_anno_{}.pkl".format(split)), "wb"))



def save_id_file(args):
    data = json.load(open(args.annotation, 'r'))
    ids = {"train": [], "val": [], "test": []}
    for k, v in data.items():
        split = k
        examples = data[split]
        for i in range(len(examples)):
            img_id = examples[i]['id']
            ids[split].append(img_id)

    for split, _ids in ids.items():
        with open(os.path.join(args.save_path, "{}_ids.txt".format(split)), "w") as fout:
            for imgid in _ids:
                fout.write("{}\n".format(imgid))
def save_split_json_file(args):
    split_data = {  "train": {"images": [], "annotations": []},
                    "val": {"images": [], "annotations": []},
                    "test": {"images": [], "annotations": []},
                    }
    data = json.load(open(args.annotation, 'r'))

    for k, v in data.items():
        split = k
        examples = data[split]
        for i in range(len(examples)):
            report = clean_report_iu_xray(examples[i]['report'])
            new_image = {
                "id": examples[i]['id']
            }
            new_caption = {
                "image_id": examples[i]['id'],
                "id": 1,
                "caption": report
            }
            split_data[split]["images"].append(new_image)
            split_data[split]["annotations"].append(new_caption)
    for split, data in split_data.items():
        if split == "train":
            continue
        json.dump(data, open(os.path.join(args.save_path, "captions_{}_cocostyle.json".format(split)), "w"))

def main(args):
    # iu_caption_anno_val = pickle.load(open(os.path.join(args.save_path, "iu_caption_anno_val.pkl"), 'rb'),
    #                                   encoding='bytes')
    # iu_train_gts = pickle.load(open(os.path.join(args.save_path, "iu_train_gts.pkl"), 'rb'),
    #                                   encoding='bytes')
    # iu_train_cider = pickle.load(open(os.path.join(args.save_path, "iu_train_cider.pkl"), 'rb'),
    #                                   encoding='bytes')
    seed(123)
    vocab = build_vocab(args.annotation, threshold=args.threshold)
    print(len(vocab['word2idx']))
    with open(os.path.join(args.save_path, "iu_vocabulary.txt"), "w") as fout:
        for w in vocab['word2idx']:
            fout.write("{}\n".format(w))
    cap2idx(args, vocab=vocab)
    save_id_file(args)
    # iu_caption_anno_val = pickle.load(open(os.path.join(args.save_path, "iu_caption_anno_val.pkl"), 'rb'),
    #                                   encoding='bytes')
    # iu_train_gts = pickle.load(open(os.path.join(args.save_path, "iu_train_gts.pkl"), 'rb'),
    #                                   encoding='bytes')
    save_split_json_file(args)
    print('finish!')
    # get_mlc_label(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation', type=str,
                        default='/home/zgs/X-ray/iu_xray/annotation.json',
                        help='path for train annotation file')
    parser.add_argument('--semic', type=str,
                        default='/home/zgs/X-ray/iu_xray/new_data/iu_semiclabel.pkl',
                        help='path for train annotation file')
    parser.add_argument('--save_path', type=str, default='/home/zgs/X-ray/iu_xray/new_data/',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=3,
                        help='minimum word count threshold')

    args = parser.parse_args()
    main(args)