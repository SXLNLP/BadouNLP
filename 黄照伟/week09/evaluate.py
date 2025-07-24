import torch
import re
import numpy as np
from collections import defaultdict
from loader import load_data



class Evaluator:
    def __init__(self,config,model,logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"],config,shuffle =False)

    def eval(self,epoch):
        self.logger.info("开始测试第 %d 轮模型效果:"%epoch)
        self.stats_dict = {"LOCATION":defaultdict(int),
                           "TIME":defaultdict(int),
                           "PERSON": defaultdict(int),
                           "ORGANIZATION":defaultdict(int)
                           }
        self.model.eval()
        for index, batch_data in enumerate(self.valid_data):
            sentences = self.valid_data.dataset.sentences[index * self.config["batch_size"]:(index+1)* self.config["batch_size"]]
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_id ,labels =batch_data
            with torch.no_grad():
                pred_results = self.model(input_id)
            self.write_stats(labels,pred_results,sentences)
        self.show_stats()
        return
    def write_stats(self,labels,pred_results,sentences):
        assert len(labels) == len(pred_results) == len(sentences)
        if not self.config["use_crf"]:
            pred_results = torch.argmax(pred_results,dim=-1)
        for true_label,pred_label,sentence in zip(labels,pred_results,sentences):
            if not self.config["use_crf"]:
                pred_label  = pred_label.cpu().detach().tolist()
            true_label = true_label.cpu().detach().tolist()
            true_entities = self.decode(sentence,true_label)
            pred_entities = self.decode(sentence,pred_label)
            for key in ["LOCATION","TIME","PERSON","ORGANIZATION"]:
                self.stats_dict[key]["正确识别"]+=len([len for ent in pred_entities[key]if ent in true_entities])
                self.stats_dict[key]["样本实体数"]+=len(true_entities[key])
                self.stats_dict[key]["识别出实体数"]+=len(pred_entities[key])

        return

    def show_stats(self):
        F1_score = []
        for key in ["LOCATION","TIME","PERSON","ORGANIZATION"]:
            precision = self.stats_dict[key]["正确识别"] / (1e-5 +self.stats_dict[key]["识别出实体数"])
            recall = self.stats_dict[key]["正确识别"] / (1e-5 +self.stats_dict[key]["样本实体数"])
            F1 = (2 * precision *recall) /(precision +recall+1e-5)
            F1_score.append(F1)
            self.logger.info("%s类实体,准确率:%f,召回率:%f,F1:%f"%(key,precision,recall,F1))
        self.logger.info("Macro-F1:%f" %np.mean(F1_score))
        correct_pred = sum([self.stats_dict[key]["正确识别"] for key in ["LOCATION","TIME","PERSON","ORGANIZATION"]])
        total_pred = sum([self.stats_dict[key]["识别出实体数"]for key in ["LOCATION","TIME","PERSON","ORGANIZATION"]])
        true_enti = sum([self.stats_dict[key]["样本实体数"]for key in ["LOCATION","TIME","PERSON","ORGANIZATION"]])
        micro_precision = correct_pred /(total_pred +1e-5)
        micro_recall = correct_pred / (true_enti + 1e-5)
        micro_f1 = (2* micro_precision * micro_recall) / (micro_precision + micro_recall +1e-5)
        self.logger.info("Micro-F1 %f" %micro_f1)
        self.logger.info("-------------------------------------")
        return
    '''
    {
      "B-LOCATION": 0,
      "B-ORGANIZATION": 1,
      "B-PERSON": 2,
      "B-TIME": 3,
      "I-LOCATION": 4,
      "I-ORGANIZATION": 5,
      "I-PERSON": 6,
      "I-TIME": 7,
      "O": 8
    }
    '''
    def decode(self,sentence,labels):
        labels = "".join([str(x) for x in labels[:len(sentence)]])
        results =  defaultdict(list)
        for location in re.finditer("(04+)",labels):
            s,e =location.span()
            results["LOCATION"].append(sentence[s:e])
        for location in re.finditer("(15+)",labels):
            s,e=location.span()
            results["ORGANIZATION"].append(sentence[s:e])
        for location in re.finditer("(26+)",labels):
            s,e = location.span()
            results["PERSON"].append(sentence[s:e])
        for location in re.finditer("(37+)",labels):
            s,e =location.span()
            results["ITEM"].append(sentence[s:e])
        return results

