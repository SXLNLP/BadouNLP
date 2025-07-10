实现电商评论的二分类（将"文本分类练习.csv"拆分为训练集和测试集），
需要看看不同的模型的效果差别
结果需要对比输出，可以用表格的形式来呈现
    
    1. 读取数据并拆分数据为训练集和测试集
    2. 分别采用三种方法得到模型（采用BERT+pooling+线性层、分词后采用BERT+LSTM、Gated CNN）并测试模型
    2.1. 采用BERT+pooling+线性层(使用huggingface的bert-base-chinese模型)
    2.2. 采用FastText
    2.3. 采用BERT+lstm
    3. 以表格的形式输出对比结果

表格见result.csv