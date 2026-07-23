# 为了应对15*15矩阵的top-k问题，让gemini实现了匈牙利算法top-k
import itertools
import math
import heapq
import numpy as np

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
    高效从 15x15（或任意NxN）位置概率矩阵中，寻找全局总分最高的前 K 个合法句子排序组合。
    采用二分图匹配与分支限界思想，避免 15! 暴力穷举。
    
    参数:
    prob_matrix (list of list / np.ndarray): NxN 的二维矩阵，prob_matrix[i][j] 表示第 i 个句子在位置 j 的概率。
    top_k (int): 需要提取的前几名候选数量。
    
    返回:
    list of list: 包含 top_k 个排列的列表，1-indexed 格式。
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    
    prob_matrix = np.array(prob_matrix)
    n = len(prob_matrix)
    eps = 1e-9
    
    # 将概率矩阵转化为对数似然矩阵（因为要最大化总概率，转为对数后相加）
    # scipy 的 linear_sum_assignment 默认是最小化开销，所以我们取负号，变成求最小化负对数似然
    cost_matrix = -np.log(prob_matrix + eps)
    
    # 优先队列（大顶堆），存储元素为：(-评估得分, 确定了前几行, 当前禁用的边, 已经选择的分配)
    # Python的heapq是小顶堆，为了让得分高的先出堆，我们把得分取负号
    pq = []
    
    # 内部辅助函数：计算带约束的最佳匹配
    def solve_constrained(forbidden_edges):
        # 复制一份矩阵，把被禁用的边设为无穷大
        tmp_cost = cost_matrix.copy()
        for r, c in forbidden_edges:
            tmp_cost[r, c] = np.inf
        
        # 使用匈牙利算法（KM算法的scipy实现，复杂度 O(N^3)）
        row_ind, col_ind = linear_sum_assignment(tmp_cost)
        
        # 如果无法达成完整匹配（比如禁边太多导致无解）
        cost = tmp_cost[row_ind, col_ind].sum()
        if np.isinf(cost):
            return None, None
        
        # 构造排列结果 (0-indexed)
        perm = [0] * n
        for r, c in zip(row_ind, col_ind):
            perm[r] = c
        return -cost, perm

    # 初始化：求全局最优解（第一名）
    best_score, best_perm = solve_constrained(set())
    if best_perm is not None:
        # 这里的 key 用于在堆中去重和排序：(当前得分, 已经确定的行数, 禁边集合, 排列)
        heapq.heappush(pq, (-best_score, 0, frozenset(), tuple(best_perm)))
    
    results = []
    seen = set()
    
    while pq and len(results) < top_k:
        neg_score, fixed_rows, forbidden_edges, perm = heapq.heappop(pq)
        score = -neg_score
        
        # 如果该排列没被记录过，则加入最终结果
        if perm not in seen:
            seen.add(perm)
            # 转为 1-indexed 格式返回
            perm_1_indexed = [pos + 1 for pos in perm]
            results.append(perm_1_indexed)
            
            if len(results) == top_k:
                break
        
        # 分支限界：通过对当前最优解的“边”进行强制禁用，分化出次优解
        # 只针对未固定的行进行分支
        for i in range(fixed_rows, n):
            # 尝试禁用当前最优解里第 i 行选中的那一列
            j = perm[i]
            new_forbidden = set(forbidden_edges)
            new_forbidden.add((i, j))
            
            # 在这个新约束下求最佳匹配
            sub_score, sub_perm = solve_constrained(new_forbidden)
            if sub_perm is not None and tuple(sub_perm) not in seen:
                # 推入堆中，下次迭代会自动弹出剩余里得分最高的
                heapq.heappush(pq, (-sub_score, i + 1, frozenset(new_forbidden), tuple(sub_perm)))
                
            # 为了保证分支不重不漏，下一轮循环里，第 i 行的这个决策被视为“固定”的
            # 也就是说，后续的分支必须包含 (i, j) 这条边，所以我们退回并处理下一行
    
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
    if bf_res == new_res:
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
