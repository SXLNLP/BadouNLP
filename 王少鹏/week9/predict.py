
from loader import DataGenerator
from config import Config
from model import TorchModel
from transformers import BertTokenizer
import torch
from collections import defaultdict
import re
"""
测试模型效果
"""
class Predictor:
    def __init__(self,model_path, config):
        self.dg = DataGenerator(config["train_data_path"], config)
        self.model = TorchModel(config)
        self.model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
        self.model.eval()  # 设置模型为评估模式
        self.config = config
    
    def predict(self, text):
        input_id = self.dg.encode_sentence(text)
        input_tensor = torch.LongTensor(input_id).unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_tensor)
        if self.config["use_crf"]:
            pred_results = self.model.crf_layer.decode(output)
        else:
            pred_results = torch.argmax(output, dim=-1)
        return self.decode(text, pred_results[0].tolist())
    def decode(self, sentence, labels):
        labels = "".join([str(x) for x in labels[:len(sentence)]])
        results = defaultdict(list)
        for location in re.finditer("(04+)", labels):
            s, e = location.span()
            results["LOCATION"].append(sentence[s:e])
        for location in re.finditer("(15+)", labels):
            s, e = location.span()
            results["ORGANIZATION"].append(sentence[s:e])
        for location in re.finditer("(26+)", labels):
            s, e = location.span()
            results["PERSON"].append(sentence[s:e])
        for location in re.finditer("(37+)", labels):
            s, e = location.span()
            results["TIME"].append(sentence[s:e])
        return results


if __name__ == "__main__":
    text = "小明每天早上八点钟做地铁去上海三菱电梯的喜马拉雅工作室工作"
    predictor = Predictor("ner_wrok/model_output/epoch_15.pth", Config)
    pred_re = predictor.predict(text)
    print("预测结果：")
    for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
        if pred_re[key]:
            print(f"{key}: {', '.join(pred_re[key])}")
        else:
            print(f"{key}: 无")
    # 输出预测结果
    # 例如：LOCATION: 上海, ORGANIZATION: 上海三菱电梯, PERSON: 小明, TIME: 每天早上八点钟
