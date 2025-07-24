# evaluate.py
import time
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')  # 使用非交互式后端


def evaluate_classifier(classifier, X_val, y_val):
    """评估分类器性能"""
    # 预测验证集
    y_pred = classifier.predict(X_val)

    # 计算准确率
    accuracy = accuracy_score(y_val, y_pred)

    # 生成分类报告
    report = classification_report(y_val, y_pred, target_names=['差评', '好评'])

    print("\n分类器评估结果:")
    print(f"准确率: {accuracy:.4f}")
    print(report)

    return accuracy, report


def plot_evaluation(accuracy):
    """绘制评估结果"""
    try:
        plt.figure(figsize=(6, 4))
        plt.bar(['准确率'], [accuracy], color='blue')
        plt.ylim(0, 1.05)
        plt.title('分类器性能评估')
        plt.ylabel('分数')
        plt.text(0, accuracy + 0.02, f"{accuracy:.4f}", ha='center')
        plt.savefig('evaluation_result.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("评估结果图已保存至 evaluation_result.png")
    except Exception as e:
        print(f"绘图失败: {e}")


def predict_sentiment(text, vector_model, classifier):
    """预测文本情感"""
    try:
        # 预处理文本
        from loader import TextDataset
        cleaner = TextDataset()
        cleaned_text = cleaner.clean_text(text)
        tokens = cleaner.tokenize(cleaned_text)

        # 转换为向量
        vector = cleaner.text_to_vector(tokens, vector_model)

        # 预测
        prob = classifier.predict_proba([vector])[0]
        sentiment = "好评" if np.argmax(prob) == 1 else "差评"
        confidence = prob[1] if sentiment == "好评" else prob[0]

        return sentiment, confidence, max(prob)

    except Exception as e:
        print(f"预测失败: {e}")
        return "未知", 0.5, 0.5


def test_prediction_speed(texts, vector_model, classifier):
    """测试预测速度"""
    start_time = time.perf_counter()

    # 预处理和预测所有文本
    from loader import TextDataset
    cleaner = TextDataset()

    for text in texts:
        cleaned_text = cleaner.clean_text(text)
        tokens = cleaner.tokenize(cleaned_text)
        vector = cleaner.text_to_vector(tokens, vector_model)
        _ = classifier.predict([vector])

    total_time = time.perf_counter() - start_time
    num_samples = len(texts)
    speed_per_sample = (total_time / num_samples) * 1000  # 毫秒/样本

    return total_time, speed_per_sample