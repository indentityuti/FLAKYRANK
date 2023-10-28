import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import confusion_matrix

# 加载数据
data_path = '../../dataflagg/FlkeFeatures.txt'
data_all = np.loadtxt(data_path, delimiter=',', dtype=float)

# 提取特征和标签
X = data_all[:, 1:]
y = data_all[:, 0]

# 创建支持向量机分类器
svm_classifier = SVC(kernel='linear', random_state=42)  # 选择线性核，可以根据需要选择其他核函数

# 使用K折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
y_pred = cross_val_predict(svm_classifier, X, y, cv=kf)

# 训练模型
svm_classifier.fit(X, y)

# 评估模型性能
accuracy = accuracy_score(y, y_pred)
classification_rep = classification_report(y, y_pred)

print(f"准确度: {accuracy:.5f}")
print("分类报告:\n", classification_rep)

# 计算混淆矩阵
conf_matrix = confusion_matrix(y, y_pred)

# 从混淆矩阵中提取 TP、FP、TN、FN 的值
TP = conf_matrix[1, 1]  # True Positives
FP = conf_matrix[0, 1]  # False Positives
TN = conf_matrix[0, 0]  # True Negatives
FN = conf_matrix[1, 0]  # False Negatives

print(f"True Positives (TP): {TP}")
print(f"False Positives (FP): {FP}")
print(f"True Negatives (TN): {TN}")
print(f"False Negatives (FN): {FN}")

# 保存预测结果和真实标签到文件
result_folder = './result'
result_file = os.path.splitext(os.path.basename(data_path))[0]
results_path = f'./result/{result_file}.txt'
with open(results_path, 'w') as results_file:
    for true_label, prediction in zip(y, y_pred):
        results_file.write(f" {int(true_label)},  {prediction}\n")

# 保存模型性能结果到文件
performance_results_path = f'./result/{result_file}_pro.txt'
with open(performance_results_path, 'w') as performance_file:
    performance_file.write(f"准确度: {accuracy:.5f}\n")
    performance_file.write("分类报告:\n" + classification_rep + "\n")
    performance_file.write(f"True Positives (TP): {TP}\n")
    performance_file.write(f"False Positives (FP): {FP}\n")
    performance_file.write(f"True Negatives (TN): {TN}\n")
    performance_file.write(f"False Negatives (FN): {FN}\n")
