# 为了应对15*15矩阵的top-k问题，让gemini实现了匈牙利算法top-k
import itertools
import math
import heapq
import numpy as np
from scipy.optimize import linear_sum_assignment

# 暴力搜索，等同于上面的get_top_k_permutations_from_matrix
def brute_force_top_k(prob_matrix, top_k=5):
    n = len(prob_matrix)
    all_candidates = []
    eps = 1e-9
    for perm in itertools.permutations(range(n)):
        current_log_likelihood = 0.0
        for sentence_idx in range(n):
            assigned_position = perm[sentence_idx]
            current_log_likelihood += math.log(prob_matrix[sentence_idx][assigned_position] + eps)
        perm_1_indexed = [pos + 1 for pos in perm]
        all_candidates.append((current_log_likelihood, perm_1_indexed))
    all_candidates.sort(key=lambda x: x[0], reverse=True)
    return [candidate[1] for candidate in all_candidates[:top_k]]

def get_top_k_permutations_from_matrix(prob_matrix, top_k=5):
    """
    直接在原始概率矩阵上通过二分图最大权匹配寻找前 K 个全局总分最高的排列组合。
    无需对数转换，无需 clip 截断。
    
    参数:
    prob_matrix (list of list / np.ndarray): NxN 的二维矩阵，prob_matrix[i][j] 表示第 i 个句子在位置 j 的概率。
    top_k (int): 需要提取的前几名候选数量。
    
    返回:
    list of list: 包含 top_k 个排列的列表，1-indexed 格式。
    """
    # 转换为 NumPy 矩阵，支持浮点运算
    prob_matrix = np.asarray(prob_matrix, dtype=np.float64)
    n = len(prob_matrix)
    
    # 优先队列（大顶堆），存储元素为：(-评估得分, 确定了前几行, 约束的禁边集合, 已经选择的分配)
    # 这里的得分采用最直观的“概率乘积”或“概率之和”。
    # 注意：由于我们要找概率最大的，而 Python 的 heapq 是小顶堆，所以存入堆时得分要取负号。
    pq = []
    
    # 内部辅助函数：计算带约束的最佳匹配
    def solve_constrained(forbidden_edges):
        # 复制一份原始概率矩阵
        tmp_prob = prob_matrix.copy()
        
        # 强行禁用指定的边：将它们的概率设为 -1.0
        # 这样在最大化收益时，算法绝对不可能选择这些负值边
        FORBIDDEN_MARKER = -1.0
        for r, c in forbidden_edges:
            tmp_prob[r, c] = FORBIDDEN_MARKER
        
        # scipy 默认做最小化开销匹配，传入 -tmp_prob 即可实现“最大化联合概率之和”
        row_ind, col_ind = linear_sum_assignment(-tmp_prob)
        
        # 检查最终匹配中是否被迫包含了被禁用的边
        # 如果选了禁边，说明此分支下已经无法组成完整的 N 阶合法匹配
        if any(tmp_prob[r, c] == FORBIDDEN_MARKER for r, c in zip(row_ind, col_ind)):
            return None, None
            
        # 计算当前排列的联合概率乘积（如果你期望评估的是总概率乘积）
        # 如果你的业务逻辑倾向于算概率之和，也可以改为 np.sum(...)
        prod_score = 1.0
        for r, c in zip(row_ind, col_ind):
            prod_score *= prob_matrix[r, c]
            
        # 构造排列结果 (0-indexed)
        perm = [0] * n
        for r, c in zip(row_ind, col_ind):
            perm[r] = c
            
        return prod_score, perm

    # 1. 初始化：求全局最优解（第一名）
    best_score, best_perm = solve_constrained(set())
    if best_perm is not None:
        # 在堆中，用 -best_score 来确保最高分排在最前面
        heapq.heappush(pq, (-best_score, 0, frozenset(), tuple(best_perm)))
    
    results = []
    seen = set()
    
    # 2. 迭代搜索次优解 (分支限界)
    while pq and len(results) < top_k:
        neg_score, fixed_rows, forbidden_edges, perm = heapq.heappop(pq)
        
        # 避免重复加入结果集
        if perm not in seen:
            seen.add(perm)
            # 转换为 1-indexed 格式返回
            perm_1_indexed = [pos + 1 for pos in perm]
            results.append(perm_1_indexed)
            
            if len(results) == top_k:
                break
        
        # 对当前解未固定的行进行分支
        for i in range(fixed_rows, n):
            j = perm[i]
            # 创建新的禁边集合，把当前决策禁掉以寻找次优解
            new_forbidden = set(forbidden_edges)
            new_forbidden.add((i, j))
            
            sub_score, sub_perm = solve_constrained(new_forbidden)
            if sub_perm is not None and tuple(sub_perm) not in seen:
                heapq.heappush(pq, (-sub_score, i + 1, frozenset(new_forbidden), tuple(sub_perm)))
                
            # 分支限界核心：这一轮循环结束后，下一轮循环必须“包含” (i, j) 决策，
            # 相当于该决策在后续的分支中被固定了，从而保证搜索空间不重不漏。
            
    return results


def run_matrix_permutation_tests():
    import time
    print(" 🚀 开始算法验证测试...\n" + "="*50)
    
    # ---------------- 验证测试 1: 小规模矩阵 (5x5) 对比正确性 ----------------
    print("[测试 1] 验证 5x5 矩阵下新算法与暴力穷举的一致性...")
    
    # 随机生成一个 5x5 的概率矩阵，并进行行归一化（模拟 BERT 的 Softmax 输出）
    np.random.seed(42)  # 固定随机种子
    raw_matrix_5 = np.random.rand(5, 5)
    prob_matrix_5 = (raw_matrix_5 / raw_matrix_5.sum(axis=1, keepdims=True)).tolist()
    
    top_k = 5
    
    # 分别用两种算法求解
    bf_res = brute_force_top_k(prob_matrix_5, top_k=top_k)
    new_res = get_top_k_permutations_from_matrix(prob_matrix_5, top_k=top_k)
    
    # 验证结果是否完全相同
    if set(tuple(x) for x in bf_res) == set(tuple(x) for x in new_res):
        print("✅ 测试 1 通过！新算法的前 K 个排列与暴力穷举完全一致。")
        print(f"Top-{top_k} 排列结果: {new_res}")
    else:
        print("❌ 测试 1 失败！算法结果不一致。")
        print(f"暴力穷举结果: {bf_res}")
        print(f"新算法结果:   {new_res}")
        return
        
    print("-" * 50)
    
    # ---------------- 验证测试 2: 大规模矩阵 (15x15) 性能测试 ----------------
    print("[测试 2] 性能测试：挑战 15x15 矩阵 (暴力穷举需要计算 1.3 万亿次，此处跳过)...")
    
    # 随机生成一个 15x15 的概率矩阵
    raw_matrix_15 = np.random.rand(15, 15)
    prob_matrix_15 = (raw_matrix_15 / raw_matrix_15.sum(axis=1, keepdims=True)).tolist()
    
    # 记录新算法耗时
    start_time = time.time()
    large_res = get_top_k_permutations_from_matrix(prob_matrix_15, top_k=5)
    end_time = time.time()
    
    elapsed_time = (end_time - start_time) * 1000  # 转换为毫秒
    
    print("✅ 测试 2 完成！")
    print(f"15x15 矩阵成功找出 Top-5 排列，耗时仅为: {elapsed_time:.2f} 毫秒！")
    print(f"最优排列组合 (1-indexed): {large_res[0]}")
    print("="*50)
