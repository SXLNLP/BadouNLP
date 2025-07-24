# -*- coding: utf-8 -*-
import torch


class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.stats_dict = {"correct": 0, "wrong": 0}

    def eval(self, valid_data, epoch):
        self.logger.info(f"Evaluating epoch {epoch}")
        self.model.eval()
        self.stats_dict = {"correct": 0, "wrong": 0}

        for index, batch_data in enumerate(valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]

            input_ids, labels = batch_data
            with torch.no_grad():
                pred_results = self.model(input_ids)
            self.write_stats(labels, pred_results)

        acc = self.show_stats()
        return acc

    def write_stats(self, labels, pred_results):
        assert len(labels) == len(pred_results)
        pred_labels = torch.argmax(pred_results, dim=-1)

        for i in range(len(labels)):
            if int(labels[i]) == int(pred_labels[i]):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1

    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        total = correct + wrong
        self.logger.info(f"Evaluation samples: {total}")
        self.logger.info(f"Correct: {correct}, Wrong: {wrong}")
        self.logger.info(f"Accuracy: {correct / total:.4f}")
        self.logger.info("-" * 20)
        return correct / total