import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import os

data_folder = './jiajia'  # 数据文件夹路径
data_files = []

# 遍历数据文件夹下的所有文件
for filename in os.listdir(data_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(data_folder, filename)
        data_files.append(file_path)

# 创建一个文件夹用于保存结果
result_folder = './results'
os.makedirs(result_folder, exist_ok=True)

for test_file in data_files:
    # 加载测试数据
    data_test = np.loadtxt(test_file, delimiter=',', dtype=float)

    # 提取测试特征和标签
    X_test = data_test[:, 1:]
    y_test = data_test[:, 0]

    # 创建随机森林分类器
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # 初始化变量用于保存训练集的数据
    X_train_all = []
    y_train_all = []

    # 构建训练集
    for train_file in data_files:
        if train_file != test_file:
            # 加载训练数据
            data_train = np.loadtxt(train_file, delimiter=',', dtype=float)

            # 提取训练特征和标签
            X_train = data_train[:, 1:]
            y_train = data_train[:, 0]

            # 使用过采样方法平衡训练数据集
            oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
            X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

            X_train_all.append(X_train_resampled)
            y_train_all.append(y_train_resampled)

    # 将所有训练集合并
    X_train_combined = np.vstack(X_train_all)
    y_train_combined = np.hstack(y_train_all)

    # 训练模型
    rf_classifier.fit(X_train_combined, y_train_combined)

    # 进行预测
    y_pred = rf_classifier.predict(X_test)

    # 评估模型性能
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    result_file = os.path.join(result_folder, os.path.splitext(os.path.basename(test_file))[0])
    print(f"{result_file}准确度: {accuracy:.2f}")
    print(f"{result_file}分类报告:\n", classification_rep)

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pred)

    # 从混淆矩阵中提取 TP、FP、TN、FN 的值
    TP = conf_matrix[1, 1]  # True Positives
    FP = conf_matrix[0, 1]  # False Positives
    TN = conf_matrix[0, 0]  # True Negatives
    FN = conf_matrix[1, 0]  # False Negatives

    print(f"{result_file}True Positives (TP): {TP}")
    print(f"{result_file}False Positives (FP): {FP}")
    print(f"{result_file}True Negatives (TN): {TN}")
    print(f"{result_file}False Negatives (FN): {FN}")

    y_prob = rf_classifier.predict_proba(X_test)[:, 1]

    # 保存得分到文件夹中，以测试集名称命名
    result_file = os.path.join(result_folder, os.path.splitext(os.path.basename(test_file))[0])
    np.savetxt(f'{result_file}_presocer.txt', y_prob, delimiter=',', fmt='%.4f')  # 保存测试集的真实标签到文件
    np.savetxt(f'{result_file}_labeltest.txt', y_test, delimiter=',', fmt='%d')
