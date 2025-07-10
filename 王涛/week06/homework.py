from transformers import BertModel


bert = BertModel.from_pretrained(r"D:\models\google-bert\bert-base-chinese")
state_dict = bert.state_dict()
words_size = 21128
hidden_size = 768
position_size = 512
token_size = 2
# embeddings层
embeddings_size = words_size*hidden_size +\
                  position_size*hidden_size  +\
                  token_size*hidden_size +\
                  hidden_size + hidden_size

size1 = state_dict['embeddings.word_embeddings.weight'].numpy().size +\
        state_dict['embeddings.position_embeddings.weight'].numpy().size +\
        state_dict['embeddings.token_type_embeddings.weight'].numpy().size +\
        state_dict['embeddings.LayerNorm.weight'].numpy().size +\
        state_dict['embeddings.LayerNorm.bias'].numpy().size

# encoder 自注意力和前馈神经网络
num_layer = 12
num_head = 12
#
encoder_size = state_dict['encoder.layer.0.attention.self.query.weight'].numpy().size +\
        state_dict['encoder.layer.0.attention.self.query.bias'].numpy().size +\
        state_dict['encoder.layer.0.attention.self.key.weight'].numpy().size +\
        state_dict['encoder.layer.0.attention.self.key.bias'].numpy().size +\
        state_dict['encoder.layer.0.attention.self.value.weight'].numpy().size +\
        state_dict['encoder.layer.0.attention.self.value.bias'].numpy().size +\
        state_dict['encoder.layer.0.attention.output.dense.weight'].numpy().size +\
        state_dict['encoder.layer.0.attention.output.dense.bias'].numpy().size +\
        state_dict['encoder.layer.0.attention.output.LayerNorm.weight'].numpy().size +\
        state_dict['encoder.layer.0.attention.output.LayerNorm.bias'].numpy().size +\
        state_dict['encoder.layer.0.intermediate.dense.weight'].numpy().size +\
        state_dict['encoder.layer.0.intermediate.dense.bias'].numpy().size +\
        state_dict['encoder.layer.0.output.dense.weight'].numpy().size +\
        state_dict['encoder.layer.0.output.dense.bias'].numpy().size +\
        state_dict['encoder.layer.0.output.LayerNorm.weight'].numpy().size +\
        state_dict['encoder.layer.0.output.LayerNorm.bias'].numpy().size
encoder_size *=12

# pooler层
pooler_size = state_dict['pooler.dense.weight'].numpy().size +\
        state_dict['pooler.dense.bias'].numpy().size


print(embeddings_size," ",size1," ",encoder_size," ",pooler_size)
