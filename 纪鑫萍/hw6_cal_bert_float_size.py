from transformers import BertModel

bert = BertModel.from_pretrained(r"/Users/juju/Downloads/bert-base-chinese", return_dict=False)
state_dict = bert.state_dict()


def cal_weight(state_dict, bert_config):
    layer_num = bert_config.num_hidden_layers
    hidden_size = bert_config.hidden_size
    intermediate_size = bert_config.intermediate_size
    vocab_size = bert_config.vocab_size
    type_vocab_size = bert_config.type_vocab_size
    max_position_embeddings = bert_config.max_position_embeddings

    """
    1、根据config参数手动记录每层shape（练习）
    """
    shape = {
        # embedding层
        "embeddings.word_embeddings.weight": (vocab_size, hidden_size),
        "embeddings.token_type_embeddings.weight": (type_vocab_size, hidden_size),
        "embeddings.position_embeddings.weight": (max_position_embeddings, hidden_size),
        "embeddings.LayerNorm.weight": (hidden_size,),  # ?
        "embeddings.LayerNorm.bias": (hidden_size,)  # ?
    }
    for i in range(layer_num):
        shape.update(
            {
                # encoder - attention层
                "encoder.layer.%d.attention.self.query.weight" % i: (hidden_size, hidden_size),
                "encoder.layer.%d.attention.self.query.bias" % i: (hidden_size,),  # ?
                "encoder.layer.%d.attention.self.key.weight" % i: (hidden_size, hidden_size),
                "encoder.layer.%d.attention.self.key.bias" % i: (hidden_size,),
                "encoder.layer.%d.attention.self.value.weight" % i: (hidden_size, hidden_size),
                "encoder.layer.%d.attention.self.value.bias" % i: (hidden_size,),
                "encoder.layer.%d.attention.output.dense.weight" % i: (hidden_size, hidden_size),
                "encoder.layer.%d.attention.output.dense.bias" % i: (hidden_size,),
                "encoder.layer.%d.attention.output.LayerNorm.weight" % i: (hidden_size,),
                "encoder.layer.%d.attention.output.LayerNorm.bias" % i: (hidden_size,),
                # encoder - feed forward
                "encoder.layer.%d.intermediate.dense.weight" % i: (intermediate_size, hidden_size),
                # intermediate_size=4*hidden_size，为何是行数4倍？因为线性层公式是y = x*A.T + b ======要转置
                "encoder.layer.%d.intermediate.dense.bias" % i: (intermediate_size,),  # 为何bias是行数？
                "encoder.layer.%d.output.dense.weight" % i: (hidden_size, intermediate_size),  # ======要转置
                "encoder.layer.%d.output.dense.bias" % i: (hidden_size,),
                "encoder.layer.%d.output.LayerNorm.weight" % i: (hidden_size,),
                "encoder.layer.%d.output.LayerNorm.bias" % i: (hidden_size,),
            }
        )

    """
    2、根据bert所有层参数核验shape信息并累加浮点数个数
    """
    embedding_float_count, attention_float_count, feed_forward_float_count = 0, 0, 0
    for name, value in state_dict.items():
        # 输出bert和计算的信息
        # print("【", name, "】--bert-size--", value.shape, "--cal-size--", shape.get(name), "--是否相等--",
        #       value.shape == shape.get(name))
        # 不一致的取出调整
        if value.shape != shape.get(name):
            print(f"不一致层【{name}】===bert-size={value.shape}，cal-size={shape.get(name)}")

        if name.startswith("embeddings"):  # embedding层浮点数个数计算
            embedding_float_count += shape.get(name)[0] * (
                shape.get(name)[1] if (len(shape.get(name)) > 1 and shape.get(name)[1]) else 1)

        elif "attention" in name:  # attention层浮点数个数计算
            attention_float_count += shape.get(name)[0] * (
                shape.get(name)[1] if (len(shape.get(name)) > 1 and shape.get(name)[1]) else 1)

        elif "pooler" not in name:  # attention层浮点数个数
            feed_forward_float_count += shape.get(name)[0] * (
                shape.get(name)[1] if (len(shape.get(name)) > 1 and shape.get(name)[1]) else 1)

    """
    3、根据config参数进行公式计算【作业部分】
    """
    cal_embedding_float_count = vocab_size * hidden_size + type_vocab_size * hidden_size + max_position_embeddings * hidden_size
    cal_embedding_float_count += hidden_size * 2  # layerNorm层的权重和偏置

    cal_attention_float_count = (hidden_size * hidden_size * 3  # QKV的权重
                                 + hidden_size * 3  # QKV的偏置
                                 + hidden_size * hidden_size + hidden_size  # QKV之后的线性层的权重和偏置
                                 + hidden_size * 2)  # layerNorm层的权重和偏置
    cal_attention_float_count *= layer_num  # 多层

    cal_feed_forward_float_count = (intermediate_size * hidden_size  # 第一层线性层的权重
                                    + hidden_size * intermediate_size  # 第二层线性层的权重
                                    + intermediate_size + hidden_size)  # 两层偏置
    cal_feed_forward_float_count += hidden_size * 2  # layerNorm层的权重和偏置
    cal_feed_forward_float_count *= layer_num  # 多层

    """
    4、输出结果
    """
    print(f"计算的embedding层浮点数个数==={cal_embedding_float_count}, "
          f"记录的embedding层浮点数个数==={embedding_float_count}, "
          f"公式计算是否准确==={cal_embedding_float_count == embedding_float_count}")
    print(f"计算的attention层浮点数个数==={cal_attention_float_count}, "
          f"记录的attention层浮点数个数==={attention_float_count}, "
          f"公式计算是否准确==={cal_attention_float_count == attention_float_count}")
    print(f"计算的ffn层浮点数个数==={cal_feed_forward_float_count}, "
          f"记录的ffn层浮点数个数==={feed_forward_float_count}, "
          f"公式计算是否准确==={cal_feed_forward_float_count == feed_forward_float_count}")


if __name__ == '__main__':
    cal_weight(state_dict, bert.config)
