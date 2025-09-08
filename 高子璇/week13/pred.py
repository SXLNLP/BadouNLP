import torch
import logging
from model import TorchModel
from peft import get_peft_model, LoraConfig
from evaluate import Evaluator
from config import Config
from loader import load_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 大模型微调策略
tuning_tactics = Config["tuning_tactics"]

print("正在使用 %s" % tuning_tactics)

if tuning_tactics == "lora_tuning":
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "key", "value"]
    )
# 其他微调策略可以根据需要添加

# 重建模型
model = TorchModel
model = get_peft_model(model, peft_config)

# 加载微调部分权重
if tuning_tactics == "lora_tuning":
    loaded_weight = torch.load('output/lora_tuning.pth')
# 其他微调策略的权重加载可以根据需要添加

# 更新模型权重
model.load_state_dict(loaded_weight, strict=False)

# 将模型迁移到 GPU（如果可用）
if torch.cuda.is_available():
    model = model.cuda()

# 进行一次测试
evaluator = Evaluator(Config, model, logger)
evaluator.eval(0)

# 预测函数
def predict(text):
    # 加载数据
    data_generator = DataGenerator("data/predict.json", Config)
    data_loader = DataLoader(data_generator, batch_size=1, shuffle=False)

    # 准备输入数据
    input_ids = data_generator.encode_sentence(text)
    input_ids = torch.LongTensor([input_ids])
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    # 模型预测
    model.eval()
    with torch.no_grad():
        output = model(input_ids)[0]
        pred_labels = torch.argmax(output, dim=2).cpu().numpy().flatten()

    # 将标签索引转换为标签名称
    label_to_index = {i: label for label, i in Config["ner_tags"].items()}
    pred_labels = [label_to_index[label] for label in pred_labels]

    return pred_labels

# 测试预测
if __name__ == "__main__":
    text = "张三在北京工作"
    print(f"输入文本: {text}")
    pred_labels = predict(text)
    print(f"预测标签: {pred_labels}")
