import torch
import logging
from model import TorchModel
from peft import get_peft_model, LoraConfig, PromptTuningConfig, PrefixTuningConfig, PromptEncoderConfig

from evaluate import Evaluator
from config import Config


logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#大模型微调策略
tuning_tactics = Config["tuning_tactics"]

print("正在使用 %s"%tuning_tactics)

if tuning_tactics == "lora_tuning":
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "key", "value"]
    )
elif tuning_tactics == "p_tuning":
    peft_config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
elif tuning_tactics == "prompt_tuning":
    peft_config = PromptTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
elif tuning_tactics == "prefix_tuning":
    peft_config = PrefixTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)

#重建模型
model = TorchModel
# print(model.state_dict().keys())
# print("====================")

model = get_peft_model(model, peft_config)
# print(model.state_dict().keys())
# print("====================")

state_dict = model.state_dict()

#将微调部分权重加载
if tuning_tactics == "lora_tuning":
    loaded_weight = torch.load('output/lora_tuning.pth')
elif tuning_tactics == "p_tuning":
    loaded_weight = torch.load('output/p_tuning.pth')
elif tuning_tactics == "prompt_tuning":
    loaded_weight = torch.load('output/prompt_tuning.pth')
elif tuning_tactics == "prefix_tuning":
    loaded_weight = torch.load('output/prefix_tuning.pth')

print(loaded_weight.keys())
state_dict.update(loaded_weight)

#权重更新后重新加载到模型
model.load_state_dict(state_dict)

#进行一次测试
model = model.cuda()
evaluator = Evaluator(Config, model, logger)
evaluator.eval(0)
