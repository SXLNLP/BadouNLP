import torch
import math
import numpy as np
from transformers import BertModel, BertConfig

class BertParameterCalculator:
    """BERT模型参数量计算器"""
    def __init__(self, model):
        self.model = model
        self.state_dict = model.state_dict()
        self.config = model.config
    
    def calculate(self):
        total_params = 0       
        emb_params = 0
        emb_params += np.prod(self.state_dict['embeddings.word_embeddings.weight'].shape)    
        emb_params += np.prod(self.state_dict['embeddings.position_embeddings.weight'].shape)    
        emb_params += np.prod(self.state_dict['embeddings.token_type_embeddings.weight'].shape)   
        emb_params += np.prod(self.state_dict['embeddings.LayerNorm.weight'].shape)
        emb_params += np.prod(self.state_dict['embeddings.LayerNorm.bias'].shape)
        
        total_params += emb_params
        print(f"1. Embedding层: {emb_params:,} 参数")

        layer_params = 0
        i = 0  
   
        attn_params = 0
        for key in ['query', 'key', 'value']:
            attn_params += np.prod(self.state_dict[f'encoder.layer.{i}.attention.self.{key}.weight'].shape)
            attn_params += np.prod(self_state_dict[f'encoder.layer.{i}.attention.self.{key}.bias'].shape)
        
        attn_params += np.prod(self.state_dict[f'encoder.layer.{i}.attention.output.dense.weight'].shape)
        attn_params += np.prod(self.state_dict[f'encoder.layer.{i}.attention.output.dense.bias'].shape)
        
        ffn_params = 0
        ffn_params += np.prod(self.state_dict[f'encoder.layer.{i}.intermediate.dense.weight'].shape)
        ffn_params += np.prod(self.state_dict[f'encoder.layer.{i}.intermediate.dense.bias'].shape)
        ffn_params += np.prod(self.state_dict[f'encoder.layer.{i}.output.dense.weight'].shape)
        ffn_params += np.prod(self.state_dict[f'encoder.layer.{i}.output.dense.bias'].shape)

        ln_params = 0
        ln_params += np.prod(self.state_dict[f'encoder.layer.{i}.attention.output.LayerNorm.weight'].shape)
        ln_params += np.prod(self.state_dict[f'encoder.layer.{i}.attention.output.LayerNorm.bias'].shape)
        ln_params += np.prod(self.state_dict[f'encoder.layer.{i}.output.LayerNorm.weight'].shape)
        ln_params += np.prod(self.state_dict[f'encoder.layer.{i}.output.LayerNorm.bias'].shape)
        
        layer_params = attn_params + ffn_params + ln_params
        total_params += layer_params * self.config.num_hidden_layers
        
        print(f"2. 单Transformer层: {layer_params:,} 参数")
        print(f"   - 注意力机制: {attn_params:,}")
        print(f"   - 前馈网络: {ffn_params:,}")
        print(f"   - 层归一化: {ln_params:,}")
        print(f"   × {self.config.num_hidden_layers}层 = {layer_params*self.config.num_hidden_layers:,}")

        pooler_params = 0
        pooler_params += np.prod(self.state_dict['pooler.dense.weight'].shape)
        pooler_params += np.prod(self.state_dict['pooler.dense.bias'].shape)
        total_params += pooler_params
        
        print(f"3. Pooler层: {pooler_params:,} 参数")
        print(f"\n总参数量: {total_params:,} (约 {total_params/1e6:.1f}M)")
        
        return total_params

class DiyBert:
    """手动实现的BERT模型"""
    def __init__(self, config, state_dict):
        self.config = config
        self.state_dict = state_dict
        self.load_weights()
    
    def load_weights(self):
        # 初始化所有权重
        self.word_emb = self.state_dict['embeddings.word_embeddings.weight'].numpy()
        self.pos_emb = self.state_dict['embeddings.position_embeddings.weight'].numpy()
        self.seg_emb = self.state_dict['embeddings.token_type_embeddings.weight'].numpy()
        self.emb_ln_gamma = self.state_dict['embeddings.LayerNorm.weight'].numpy()
        self.emb_ln_beta = self.state_dict['embeddings.LayerNorm.bias'].numpy()
        
        # 初始化Transformer层
        self.layers = []
        for i in range(self.config.num_hidden_layers):
            layer = {
                # 注意力权重
                'q_weight': self.state_dict[f'encoder.layer.{i}.attention.self.query.weight'].numpy(),
                'q_bias': self.state_dict[f'encoder.layer.{i}.attention.self.query.bias'].numpy(),
                'k_weight': self.state_dict[f'encoder.layer.{i}.attention.self.key.weight'].numpy(),
                'k_bias': self.state_dict[f'encoder.layer.{i}.attention.self.key.bias'].numpy(),
                'v_weight': self.state_dict[f'encoder.layer.{i}.attention.self.value.weight'].numpy(),
                'v_bias': self.state_dict[f'encoder.layer.{i}.attention.self.value.bias'].numpy(),
                'attn_out_weight': self.state_dict[f'encoder.layer.{i}.attention.output.dense.weight'].numpy(),
                'attn_out_bias': self.state_dict[f'encoder.layer.{i}.attention.output.dense.bias'].numpy(),
                'attn_ln_gamma': self.state_dict[f'encoder.layer.{i}.attention.output.LayerNorm.weight'].numpy(),
                'attn_ln_beta': self.state_dict[f'encoder.layer.{i}.attention.output.LayerNorm.bias'].numpy(),
                
                # 前馈网络
                'ffn_inter_weight': self.state_dict[f'encoder.layer.{i}.intermediate.dense.weight'].numpy(),
                'ffn_inter_bias': self.state_dict[f'encoder.layer.{i}.intermediate.dense.bias'].numpy(),
                'ffn_out_weight': self.state_dict[f'encoder.layer.{i}.output.dense.weight'].numpy(),
                'ffn_out_bias': self.state_dict[f'encoder.layer.{i}.output.dense.bias'].numpy(),
                'ffn_ln_gamma': self.state_dict[f'encoder.layer.{i}.output.LayerNorm.weight'].numpy(),
                'ffn_ln_beta': self.state_dict[f'encoder.layer.{i}.output.LayerNorm.bias'].numpy()
            }
            self.layers.append(layer)
        
        # Pooler层
        self.pooler_weight = self.state_dict['pooler.dense.weight'].numpy()
        self.pooler_bias = self.state_dict['pooler.dense.bias'].numpy()
    
    def embed(self, input_ids):
        """嵌入层"""
        # 词嵌入
        token_emb = self.word_emb[input_ids]
        
        # 位置嵌入 [0,1,2,...]
        pos_ids = np.arange(len(input_ids))
        pos_emb = self.pos_emb[pos_ids]
        
        # 段落嵌入 (全0)
        seg_emb = self.seg_emb[0]  # 单句输入
        
        # 相加并LayerNorm
        emb = token_emb + pos_emb + seg_emb
        mean = emb.mean(axis=-1, keepdims=True)
        std = emb.std(axis=-1, keepdims=True)
        emb = (emb - mean) / (std + 1e-12)
        emb = emb * self.emb_ln_gamma + self.emb_ln_beta
        
        return emb
    
    def attention(self, x, layer_idx):
        """自注意力机制"""
        layer = self.layers[layer_idx]
        
        # 计算Q/K/V
        Q = np.dot(x, layer['q_weight'].T) + layer['q_bias']
        K = np.dot(x, layer['k_weight'].T) + layer['k_bias']
        V = np.dot(x, layer['v_weight'].T) + layer['v_bias']
        
        # 多头分割
        def split_heads(tensor):
            return tensor.reshape(tensor.shape[0], self.config.num_attention_heads, -1).transpose(1, 0, 2)
        
        Q = split_heads(Q)  # [heads, seq_len, dim]
        K = split_heads(K)
        V = split_heads(V)
        
        # 注意力分数
        attn_scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(Q.shape[-1])
        attn_probs = self.softmax(attn_scores)
        
        # 注意力输出
        attn_output = np.matmul(attn_probs, V)
        attn_output = attn_output.transpose(1, 0, 2).reshape(x.shape[0], -1)
        
        # 输出投影
        attn_output = np.dot(attn_output, layer['attn_out_weight'].T) + layer['attn_out_bias']
        
        # 残差连接+LayerNorm
        x = x + attn_output
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        x = (x - mean) / (std + 1e-12)
        x = x * layer['attn_ln_gamma'] + layer['attn_ln_beta']
        
        return x
    
    def feed_forward(self, x, layer_idx):
        """前馈网络"""
        layer = self.layers[layer_idx]
        
        # 中间层
        h = np.dot(x, layer['ffn_inter_weight'].T) + layer['ffn_inter_bias']
        h = self.gelu(h)
        
        # 输出层
        h = np.dot(h, layer['ffn_out_weight'].T) + layer['ffn_out_bias']
        
        # 残差连接+LayerNorm
        x = x + h
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        x = (x - mean) / (std + 1e-12)
        x = x * layer['ffn_ln_gamma'] + layer['ffn_ln_beta']
        
        return x
    
    def transformer_layer(self, x, layer_idx):
        """完整的Transformer层"""
        x = self.attention(x, layer_idx)
        x = self.feed_forward(x, layer_idx)
        return x
    
    def pooler(self, x):
        """Pooler层（取[CLS] token处理）"""
        x = np.dot(x, self.pooler_weight.T) + self.pooler_bias
        return np.tanh(x)
    
    def forward(self, input_ids):
        """前向传播"""
        # 嵌入层
        x = self.embed(input_ids)
        
        # Transformer层
        for i in range(self.config.num_hidden_layers):
            x = self.transformer_layer(x, i)
        
        # Pooler输出（取第一个token）
        pooler_output = self.pooler(x[0])
        
        return x, pooler_output
    
    @staticmethod
    def softmax(x):
        """Softmax函数"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    @staticmethod
    def gelu(x):
        """GELU激活函数"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

# 测试代码
if __name__ == "__main__":
    # 加载预训练模型
    model_name = "bert-base-chinese"
    bert = BertModel.from_pretrained(model_name)
    bert.eval()
    
    # 计算参数量
    print("===== 参数量计算 =====")
    calculator = BertParameterCalculator(bert)
    total_params = calculator.calculate()
    
    # 准备输入
    input_ids = np.array([101, 2450, 15486, 102])  # [CLS] + 两个token + [SEP]
    
    # 原始模型输出
    print("\n===== 原始BERT输出 =====")
    with torch.no_grad():
        outputs = bert(torch.tensor([input_ids]))
        print("Sequence output shape:", outputs[0].shape)
        print("Pooler output shape:", outputs[1].shape)
    
    # 手动实现输出
    print("\n===== 手动BERT输出 =====")
    diy_bert = DiyBert(bert.config, bert.state_dict())
    diy_seq_output, diy_pooler_output = diy_bert.forward(input_ids)
    print("Sequence output shape:", diy_seq_output.shape)
    print("Pooler output shape:", diy_pooler_output.shape)
    
    # 比较输出
    print("\n===== 输出比较 =====")
    print("Pooler输出差异:", np.abs(outputs[1].numpy() - diy_pooler_output).max())
