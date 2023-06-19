__author__ = 'tylin'
from .tokenizer.ptbtokenizer import PTBTokenizer
from .bleu.bleu import Bleu
from .meteor.meteor import Meteor
from .rouge.rouge import Rouge
from .cider.cider import Cider
# from radgraph import F1RadGraph
# from f1chexbert import F1CheXbert
# import os
# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:20171'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:20171'
class COCOEvalCap:
    def __init__(self, coco, cocoRes):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.coco = coco
        self.cocoRes = cocoRes
        self.params = {'image_id': cocoRes.getImgIds()}
        # self.f1radgraph = F1RadGraph(reward_level="partial")
        # self.f1chexbert = F1CheXbert(device="cuda")

    def evaluate(self):
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.cocoRes.imgToAnns[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        # print('tokenization...')
        # tokenizer = PTBTokenizer()
        # gts  = tokenizer.tokenize(gts)
        # res = tokenizer.tokenize(res)
        gts,refs = self.scorers_pre(gts)
        res,hyps = self.scorers_pre(res)
        # accuracy, accuracy_not_averaged, class_report, class_report_5 = self.f1chexbert(hyps=hyps,refs=refs)
        # scorerad, _, hypothesis_annotation_lists, reference_annotation_lists = self.f1radgraph(hyps=hyps,refs=refs)
        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]

        # =================================================
        # Compute scores
        # =================================================
        eval = {}
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, imgIds, m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, imgIds, method)
                print("%s: %0.3f"%(method, score))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]
    def scorers_pre(self, dict_pre):
        dict = {}
        list_cap=[]
        for k, v in dict_pre.items():
            caption = v[0]['caption']
            dict[k] = [caption]
            list_cap = list_cap + [caption]
        return dict,list_cap

