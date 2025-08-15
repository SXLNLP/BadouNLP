import torch
import logging
from model import NERModel
from peft import get_peft_model, LoraConfig
from config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 大模型微调策略
tuning_tactics = Config["tuning_tactics"]
logger.info("正在使用 %s" % tuning_tactics)

# LoRA 配置
if tuning_tactics == "lora_tuning":
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "key", "value"]
    )

# 重建模型
model = NERModel().get_model()
model = get_peft_model(model, peft_config)

# 加载微调权重
loaded_weight = torch.load('output/lora_tuning.pth')
state_dict = model.state_dict()
state_dict.update(loaded_weight)
model.load_state_dict(state_dict)

# 进行一次测试
model = model.cuda()
evaluator = Evaluator(Config, model, logger)
evaluator.eval(0)
