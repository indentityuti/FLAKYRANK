import numpy as np

def average_precision(y_true, y_scores):
   
    precision = []
    num_relevant = sum(y_true)
    if num_relevant == 0:
        return 0.0

    sorted_indices = np.argsort(y_scores)[::-1]
    num_retrieved = 0
    for i, idx in enumerate(sorted_indices):
        if y_true[idx] == 1:
            num_retrieved += 1
            precision.append(num_retrieved / (i + 1))

    if not precision:
        return 0.0

    return np.mean(precision)
def mean_average_precision_at_5(y_true, y_pred):
    # 计算平均精度均值（MAP@5）
    llen=int(0.05*len(y_pred))
    sorted_indices = np.argsort(y_pred)[::-1][:llen]  # 取前五个预测结果的索引
    num_correct = 0
    total_precision = 0.0

    for i, idx in enumerate(sorted_indices):
        if y_true[idx] == 1:
            num_correct += 1
            total_precision += num_correct / (i + 1)

    if num_correct == 0:
        return 0.0

    return total_precision / num_correct

def mean_reciprocal_rank(y_true, y_pred):
    
    sorted_indices = np.argsort(y_pred)[::-1]
    for i, idx in enumerate(sorted_indices):
        if y_true[idx] == 1:
            return 1 / (i + 1)
    return 0.0

def dcg_at_k(y_true, y_pred, k):
    
    dcg = 0.0
    klen=int(k*0.01*len(y_pred))
    for i in range(klen):
        rel = y_pred[i] if y_true[i] == y_pred[i] else 0
        dcg += (2**rel - 1) / np.log2(i + 2)
    return dcg

def ndcg_at_k(y_true, y_pred, k):
    # 计算归一化折损累积增益（NDCG）
    dcg = dcg_at_k(y_true, y_pred, k)
    idcg = dcg_at_k(y_true, y_true, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def precision_at_1(y_true, y_pred):
    # 计算Precision at 1 (P@1)
    correct_prediction = y_true[0] == y_pred[0]
    return 1 if correct_prediction else 0

def precision_at_k(y_true, y_pred, k):
    # 计算Precision at k (P@k)
    k = min(k, len(y_pred))
    correct_predictions = sum(y_true[:k] == y_pred[:k])
    return correct_predictions / k

data_path='./lastresult/activiti.txt'
data=np.loadtxt(data_path, delimiter=',', dtype=float)
sorted_indices = np.argsort(data[:, 0])[::-1]
# 使用排序后的索引重新排列数组
result = data[sorted_indices]
y_true = result[:, 0].astype(int)
y_pred = result[:, 1].astype(int)
y_scores = result[:, 2]
# 计算P@3和P@5
k_3 = 3
k_5 = 5

p_at_3 = precision_at_k(y_true, y_pred, k_3)
p_at_5 = precision_at_k(y_true, y_pred, k_5)


map_score = average_precision(y_true, y_pred)
mrr_score = mean_reciprocal_rank(y_true, y_pred)
# k = int(0.05*len(data[:,0]) )
# 你可以指定一个合适的k值
m=10
k=5
j=3
ndcg_score = ndcg_at_k(y_true, y_pred, k)
ndcg_score_3 = ndcg_at_k(y_true, y_pred, j)
ndcg_score_10 = ndcg_at_k(y_true, y_pred, m)
p_at_1 = precision_at_1(y_true, y_pred)
map5=mean_average_precision_at_5(y_true, y_pred)
# print("P@1:", p_at_1)
# print(f"P@{k_3}:", p_at_3)
# print(f"P@{k_5}:", p_at_5)
# 计算MAP，MRR和NDCG
print("MAP:", map_score)
print("MAP@5:",map5)
print("MRR:", mrr_score)
print(f"NDCG@{j}:", ndcg_score_3)
print(f"NDCG@{k}:", ndcg_score)
print(f"NDCG@{m}:", ndcg_score_10)




