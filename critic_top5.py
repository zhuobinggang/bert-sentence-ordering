# 从概率矩阵中找到top5序列，然后用critic模型计算reward
from critic_bert_simple import get_critic_score, default_critic_model
import itertools
import math
from critic_bert import resort_paragraph, recover_unsorted_paragraph

def get_top_k_permutations_from_matrix(prob_matrix, top_k=5):
    """
    从 bert1 的 5x5 位置概率矩阵中，寻找全局总分最高的前 K 个合法句子排序组合。
    
    参数:
    prob_matrix (list of list): 5x5 的二维列表或 Tensor。
                                prob_matrix[i][j] 表示第 i 个输入句子被预测在位置 j (0~4) 的概率。
    top_k (int): 需要提取的前几名候选数量，默认取前 5 名。
    
    返回:
    list of list: 包含 top_k 个排列的列表。每个排列的形式为 [p0, p1, p2, p3, p4] (1-indexed)，
                  其格式与您数据集中的 predicted_label / true_label 完全一致。
    """
    n = len(prob_matrix)
    all_candidates = []
    eps = 1e-9  # 防止 log(0) 报错
    
    # 1. 穷举所有可能的合法不重复位置分配（一共 5! = 120 种）
    # perm 的结构为 (p0, p1, p2, p3, p4)，其中 perm[i] 代表第 i 个句子被分配的 0-indexed 位置
    for perm in itertools.permutations(range(n)):
        current_log_likelihood = 0.0
        
        # 2. 累加当前排列下，5个句子的联合对数概率
        for sentence_idx in range(n):
            assigned_position = perm[sentence_idx]
            # 累加第 sentence_idx 个句子去 assigned_position 位置的概率
            current_log_likelihood += math.log(prob_matrix[sentence_idx][assigned_position] + eps)
            
        # 3. 将 0-indexed 的位置转换为 1-indexed (符合您原先 labels 的 1~5 格式)
        perm_1_indexed = [pos + 1 for pos in perm]
        
        # 记录得分和对应的排列标签
        all_candidates.append((current_log_likelihood, perm_1_indexed))
        
    # 4. 按照对数概率得分从大到小进行排序
    all_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # 5. 截取前 K 个得分最高的排列
    top_k_permutations = [candidate[1] for candidate in all_candidates[:top_k]]
    
    return top_k_permutations


def run():
    critic_model = default_critic_model()
    # 这里假设 prob_matrix 是从 bert1 模型输出的 5x5 位置概率矩阵
    prob_matrix = [
        [0.1, 0.2, 0.3, 0.25, 0.15],
        [0.05, 0.3, 0.4, 0.2, 0.05],
        [0.2, 0.1, 0.25, 0.35, 0.1],
        [0.15, 0.25, 0.2, 0.3, 0.1],
        [0.3, 0.15, 0.1, 0.25, 0.2]
    ]
    
    top_k_permutations = get_top_k_permutations_from_matrix(prob_matrix)
    paragraph = ["Sentence 1", "Sentence 2", "Sentence 3", "Sentence 4", "Sentence 5"]  # 示例段落
    
    for idx, perm in enumerate(top_k_permutations):
        print(f"Permutation {idx+1}: {perm}")
        # 假设我们有一个函数可以根据 perm 获取对应的段落文本列表
        paragraph = resort_paragraph(paragraph, perm)  # 根据 perm 重新排序段落
        print(f"Resorted Paragraph: {paragraph}")
        score = get_critic_score(critic_model, paragraph)
        print(f"Critic Score for Permutation {idx+1}: {score}")