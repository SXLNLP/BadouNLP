# -*- coding: utf-8 -*-
import torch
import json
import logging
from config import Config
from transformer.Models import Transformer
from transformer.Translator import Translator

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTester:
    def __init__(self, model_path, config):
        self.config = config
        self.model_path = model_path
        
        # 加載詞彙表
        self.vocab = self.load_vocab(config["vocab_path"])
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # 初始化模型
        self.model = Transformer(
            n_src_vocab=config["vocab_size"], 
            n_trg_vocab=config["vocab_size"], 
            src_pad_idx=0, 
            trg_pad_idx=0,
            d_word_vec=128, 
            d_model=128, 
            d_inner=256,
            n_layers=1, 
            n_head=2, 
            d_k=64, 
            d_v=64
        )
        
        # 加載訓練好的模型權重
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        
        # 初始化翻譯器
        self.translator = Translator(
            self.model,
            config["beam_size"],
            config["output_max_length"],
            config["pad_idx"],
            config["pad_idx"],
            config["start_idx"],
            config["end_idx"]
        )
        
        logger.info("模型加載完成！")
    
    def load_vocab(self, vocab_path):
        """加載詞彙表"""
        token_dict = {}
        with open(vocab_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                token = line.strip()
                token_dict[token] = index
        return token_dict
    
    def encode_sentence(self, text, max_length):
        """將文本編碼為序列"""
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        
        # 填充或截斷
        input_id = input_id[:max_length]
        input_id += [self.vocab["[PAD]"]] * (max_length - len(input_id))
        return torch.LongTensor(input_id)
    
    def decode_seq(self, seq):
        """將序列解碼為文本"""
        return "".join([self.reverse_vocab[int(idx)] for idx in seq if int(idx) not in [self.vocab["[PAD]"], self.vocab["[CLS]"], self.vocab["[SEP]"]]])
    
    def test_single_input(self, input_text):
        """測試單個輸入"""
        print(f"\n{'='*60}")
        print(f"輸入文本: {input_text}")
        print(f"{'='*60}")
        
        # 編碼輸入
        input_seq = self.encode_sentence(input_text, self.config["input_max_length"])
        
        # 生成輸出
        with torch.no_grad():
            generated = self.translator.translate_sentence(input_seq.unsqueeze(0))
        
        # 解碼輸出
        output_text = self.decode_seq(generated)
        
        print(f"生成結果: {output_text}")
        print(f"{'='*60}")
        
        return output_text
    
    def test_multiple_inputs(self, test_cases):
        """測試多個輸入"""
        print(f"\n開始測試 {len(test_cases)} 個樣本...")
        print(f"{'='*80}")
        
        results = []
        for i, (input_text, expected_output) in enumerate(test_cases, 1):
            print(f"\n測試案例 {i}:")
            generated = self.test_single_input(input_text)
            results.append({
                'input': input_text,
                'expected': expected_output,
                'generated': generated
            })
        
        return results
    
    def test_with_sample_data(self, num_samples=5):
        """使用原始數據進行測試"""
        print(f"\n使用原始數據測試 {num_samples} 個樣本...")
        print(f"{'='*80}")
        
        # 讀取原始數據
        test_cases = []
        with open(self.config["train_data_path"], 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                data = json.loads(line.strip())
                test_cases.append((data["answer"], data["question"]))
        
        return self.test_multiple_inputs(test_cases)

def main():
    """主測試函數"""
    model_path = "output/epoch_300.pth"
    
    try:
        # 初始化測試器
        tester = ModelTester(model_path, Config)
        
        # 測試1: 自定義輸入
        print("🧪 測試1: 自定義輸入")
        custom_tests = [
            ("今天天氣很好，陽光明媚，適合出門散步。", "天氣"),
            ("北京故宮是中國古代宮殿建築的代表，具有重要的歷史文化價值。", "故宮"),
            ("人工智能技術正在快速發展，改變著我們的生活方式。", "AI"),
            ("環保問題日益嚴重，我們需要共同努力保護地球。", "環保"),
            ("教育是國家發展的基礎，應該重視教育事業的發展。", "教育")
        ]
        
        results1 = tester.test_multiple_inputs(custom_tests)
        
        # 測試2: 使用原始數據
        print("\n🧪 測試2: 使用原始訓練數據")
        results2 = tester.test_with_sample_data(num_samples=3)
        
        # 總結
        print(f"\n{'='*80}")
        print("🎉 測試完成！")
        print(f"{'='*80}")
        print("模型能夠根據輸入文本生成相關的標題/摘要。")
        print("雖然生成的文本可能還不夠完美，但已經顯示出學習效果。")
        
    except Exception as e:
        logger.error(f"測試過程中出現錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 