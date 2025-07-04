import time
import os

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import joblib

import process_data


# 树模型：决策树、bagging(随机森林)、boosting(GBDT)、Stacking;  SVM(不同核函数);  贝叶斯
def model_train(model_name, x_train, y_train):
    if model_name == 'bayes':
        model = GaussianNB()
    elif model_name == 'randomforest':
        model = RandomForestClassifier()
    elif model_name.startswith('svm'):
        # ['linear', 'poly', 'rbf', 'sigmoid']
        if model_name.endswith('poly'):
            model = SVC(kernel='poly', gamma='scale', random_state=42)
        elif model_name.endswith('rbf'):
            model = SVC(kernel='rbf', gamma='scale', random_state=42)
        elif model_name.endswith('sigmoid'):
            model = SVC(kernel='sigmoid', gamma='scale', random_state=42)
        else:
            model = SVC(kernel='linear', gamma='scale', random_state=42)
    elif model_name == 'decisiontree':
        model = DecisionTreeClassifier()
    elif model_name == 'gbdt':
        model = GradientBoostingClassifier()
    elif model_name == 'stacking':
        # 定义基模型
        base_models = [
            ('logistic', LogisticRegression(random_state=42)),
            ('svc', SVC(probability=True, random_state=42)),
            ('randomforest', RandomForestClassifier(random_state=42))
        ]
        # 定义元模型
        meta_model = LogisticRegression(random_state=42)
        # 创建堆叠模型
        model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)

    # 训练

    model.fit(x_train, y_train)
    if not os.path.isdir(r"save_model"):
        os.mkdir(r"save_model")

    model_path = f"./save_model/{model_name}.pkl"
    joblib.dump(model, filename=model_path)  # 保存为 .pkl 文件

def predict_evaluate(model, x_test, y_test):
    # 预测
    y_preds = model.predict(x_test)
    # 输出结果
    acc_score = accuracy_score(y_test, y_preds)
    pre_score = precision_score(y_test, y_preds)
    rec_score = recall_score(y_test, y_preds)
    f1_sc = f1_score(y_test, y_preds)

    # print("Classification Report:\n", class_report)
    return acc_score, pre_score, rec_score, f1_sc

def main():
    # 1、加载训练/测试数据
    path = r'../e-commerce_comments.csv'
    train_data_set, valid_data_set = process_data.process_segment_to_dataset(path)  # 基于字的词向量
    # train_data_set, valid_data_set = process_data.process_cut_words_to_vec(path, is_weight=1)   # 基于分词的词向量
    # 2、训练多种模型、比较
    # models = ['bayes', 'decisiontree', 'randomforest', 'gbdt', 'svm', 'svm-sigmoid', 'stacking']
    models = ['bayes', 'decisiontree', 'randomforest', 'gbdt', 'svm', 'svm-sigmoid']


    for model_name in models:
        print(f'\n-----------------------------模型{model_name}开始训练----------------------------------')
        start_time = time.time()
        for batch_idx, (data, label) in enumerate(train_data_set):  # 加载训练数据
            model_train(model_name, data, label)  # 不同的模型进行训练
        model = joblib.load(f'{model_name}.pkl')  # 加载保存的模型

        batch_num, accuracy, precision, recall, f1 = 0, 0, 0, 0, 0
        for batch_idx, (data, label) in enumerate(valid_data_set):  # 加载测试数据
            batch_num += 1
            acc_score, pre_score, rec_score, f1_sc = predict_evaluate(model, data, label)  # 预测数据并进行计算准确率、精确度、召回率、f1-score
            # print(f"Accuracy: {acc_score}, Precision: {pre_score}, Recall: {rec_score}, F1-sco: {f1_sc}")
            accuracy += acc_score
            precision += pre_score
            recall += rec_score
            f1 += f1_sc
        accuracy, precision, recall, f1 = accuracy/batch_num, precision/batch_num, recall/batch_num, f1/batch_num

        print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-sco: {f1}")
        end_time = time.time()
        execution_time = end_time - start_time
        print("Train execution Time:", execution_time)
    print('------------------模型全部训练结束--------------------')


if __name__ == '__main__':
    main()
