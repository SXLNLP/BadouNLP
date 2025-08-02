import torch
from transformers import BertTokenizer, BertForMaskedLM

# 加载预训练的BERT模型和分词器
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

def generate_text(prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    generated = []

    for _ in range(max_length - len(input_ids[0])):
        # 将最后一个token设为[mask]
        masked_input_ids = input_ids.clone()
        masked_input_ids[:, -1] = tokenizer.mask_token_id
        
        with torch.no_grad():
            outputs = model(masked_input_ids)
        
        predictions = outputs.logits
        predicted_index = torch.argmax(predictions[0, -1]).item()
        predicted_token = tokenizer.decode([predicted_index])
        
        if predicted_token == '[SEP]':
            break
        
        generated.append(predicted_token)
        input_ids = torch.cat((input_ids, torch.tensor([[predicted_index]])), dim=-1)
    
    return prompt + ''.join(generated)

prompt = "The capital of France is"
generated_text = generate_text(prompt)
print(generated_text)
