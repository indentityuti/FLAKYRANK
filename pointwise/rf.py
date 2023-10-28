import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 加载训练数据和标签
# 这里假设你已经有了训练数据X和标签Y，它们的形状应该分别为 (num_samples, num_features) 和 (num_samples,)
# 如果需要，你可以使用 train_test_split 分割数据为训练集和测试集
data_all = np.loadtxt('./dataflagg/result.txt', delimiter=',', dtype=float)
X = data_all[:, 1:]
Y = data_all[:, :1]  # 注意这里将标签数据赋给了Y

# 将标签数据转换为整数类型，如果标签是浮点数或其他类型
Y = Y.astype(int)

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 创建一个Random Forest分类器
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# 在训练集上拟合分类器
rf_classifier.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = rf_classifier.predict(X_test)

# 计算分类准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# 输出更多的分类性能指标
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)
# 保存测试集的真实标签到文件
np.savetxt('labeltest.txt', y_test, delimiter=',', fmt='%d')
