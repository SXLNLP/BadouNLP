# L=词表大小，H=768，P=位置嵌入数512，D=分段数2，A=头数
def calculate(L, H, P, D, A):
    # embedding
    e = L * H + 2 * H + 512 * H
    # 归一化
    e = e + 2 * 768
    print("e:", e)
    # self-attention Q,K,V三个矩阵分A个部分
    s = A * 3 * (768 * (768 / 12) + 768 / 12)
    # 线性层
    s = s + 768 ** 2 + 768
    # 残差和层归一化
    s = s + 768 * 2
    print("s:", s)
    # Feed Forward
    # h->4h
    f = 768 * 4 * 768 + 768 * 4
    # 4h->h
    f = f + 768 * 4 * 768 + 768
    # 残差和层归一化
    f = f + 2 * 768
    print("f:", f)
    # 12层编码器
    b12 = (s + f) * 12
    print("b12:", b12)
    # 总参数量
    total_data = e + b12
    return print("total_data:", total_data)


calculate(30522, 768, 512, 2, 12)
