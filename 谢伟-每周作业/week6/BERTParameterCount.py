#coding:utf-8

"""
计算BERT的总参数量
"""

def BERTParameterCount(max_len, hidden_size, max_position_embeddings, intermediate_size, num_hidden_layers):
    totalCount = 0
    """
    Embedding
    """
    # Token Embedding
    embeddingCount += max_len * hidden_size
    # Segment Embedding
    embeddingCount += 2 * hidden_size
    # Position Embedding
    embeddingCount += max_position_embeddings * hidden_size

    """
    transformer
    """

    """
    self-attention: multi-head
    """
    # Q K V
    transformerCount = 0
    parameterMultiHead = 3 * hidden_size * hidden_size
    # linear layer
    parameterMultiHead += hidden_size * hidden_size
    transformerCount += parameterMultiHead

    """
    Feed forward
    """
    # x4
    parameterFeedForward = hidden_size * intermediate_size
    # /4
    parameterFeedForward += intermediate_size * hidden_size
    transformerCount += parameterFeedForward

    """
    Layer normalization: there are two LN layers in every transformer 
    """
    transformerCount += 2 * (2 * hidden_size)

    # total
    totalCount = embeddingCount + transformerCount * num_hidden_layers
    return totalCount