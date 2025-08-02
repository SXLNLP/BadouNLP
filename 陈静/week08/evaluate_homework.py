# -*- coding: utf-8 -*-
import torch
from loader_homework import load_data

"""
模型效果测试
"""

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.train_data = load_data(config["train_data_path"], config)
        self.stats_dict = {"correct": 0, "wrong": 0}

    def knwb_to_vector(self):
        self.question_index_to_standard_question_index = {}
        self.question_ids = []
        
        for standard_question_index, question_ids in self.train_data.dataset.knwb.items():
            for question_id in question_ids:
                self.question_index_to_standard_question_index[len(self.question_ids)] = standard_question_index
                self.question_ids.append(question_id)
        
        with torch.no_grad():
            question_matrixs = torch.stack(self.question_ids, dim=0)
            if torch.cuda.is_available():
                question_matrixs = question_matrixs.cuda()
            self.knwb_vectors = self.model(question_matrixs)
            self.knwb_vectors = torch.nn.functional.normalize(self.knwb_vectors, dim=-1)

    def eval(self, epoch):
        self.logger.info(f"Testing model at epoch {epoch}")
        self.stats_dict = {"correct": 0, "wrong": 0}
        self.model.eval()
        self.knwb_to_vector()
        
        for batch_data in self.valid_data:
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            
            input_id, labels = batch_data
            with torch.no_grad():
                test_question_vectors = self.model(input_id)
            
            self.write_stats(test_question_vectors, labels)
        
        self.show_stats()

    def write_stats(self, test_question_vectors, labels):
        assert len(labels) == len(test_question_vectors)
        
        for test_question_vector, label in zip(test_question_vectors, labels):
            res = torch.mm(test_question_vector.unsqueeze(0), self.knwb_vectors.T)
            hit_index = int(torch.argmax(res.squeeze()))
            hit_index = self.question_index_to_standard_question_index[hit_index]
            
            if int(hit_index) == int(label):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1

    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        total = correct + wrong
        accuracy = correct / total if total > 0 else 0
        
        self.logger.info(f"Total predictions: {total}")
        self.logger.info(f"Correct: {correct}, Wrong: {wrong}")
        self.logger.info(f"Accuracy: {accuracy:.4f}")
        self.logger.info("-" * 20)
