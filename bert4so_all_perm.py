import itertools
from common import resort_paragraph, recover_unsorted_paragraph, add_one
from bert4so import *


def generate_permutations(n):
    return list(itertools.permutations(range(n)))


# 使用排序模型来给5!种排序可能性进行评分，选择得分最高的排序作为最终结果
def run():
    # 1. 生成5!种排序可能性
    permutations = generate_permutations(5)  # 假设有一个函数可以生成5!种排列

    # 2. 对每种排序进行评分
    scores = []
    for perm in permutations:
        score = score_permutation(perm)  # 假设有一个函数可以对排列进行评分
        scores.append((perm, score))

    # 3. 找到得分最高的排序
    best_permutation = max(scores, key=lambda x: x[1])[0]

    return best_permutation



def test_trained(sind=True, split='test', need_shuffle=True):
    logger = common.logging.getLogger(__name__)
    """
    自动扫描 checkpoints 文件夹，加载所有训练好的 BERT4SO 模型，
    并在指定的数据集划分（默认 test 集）上跑全量指标测试。
    """
    from pathlib import Path
    directory_path = Path("./checkpoints")
    
    # 根据前面设定的命名规则，动态组合搜索字符串
    # 例如：'critic_bert_sind' 或 'critic_bert_rocs'
    dataset_tag = 'sind' if sind else 'rocs'
    search_string = f"{dataset_tag}_listmle_rep"
    
    # 找出文件夹下所有匹配的 pth/ckpt 文件
    matching_files = [file for file in directory_path.glob(f"*{search_string}*") if file.is_file()]
    
    if not matching_files:
        print(f"❌ 未在 {directory_path} 中找到包含 '{search_string}' 的模型权重文件。")
        return

    # 加载对应的测试/验证数据
    test_paragraphs = sind_paragraphs(split) if sind else rocs.dataset_get()[split]
    print(f"🔍 找到 {len(matching_files)} 个匹配的模型，开始在 {dataset_tag.upper()} 的 【{split}】 集上进行测试...")

    taus = []
    accs = []
    pmrs = []
    for file in matching_files:
        # 1. 必须实例化完整的神经网络架构（包含 BERT 和 Linear Head）
        model = CriticBert()
        load_checkpoint(model, str(file))
        model.to(DEVICE)
        model.eval()
        
        total_tau = 0.0
        total_acc = 0.0
        total_pmr = 0.0
        total_count = 0
        
        # 2. 遍历测试集进行微观打分与排序还原
        for tgt_paragraph in tqdm(test_paragraphs, desc=f"Testing {file.name}"):
            if len(tgt_paragraph) <= 1:
                continue
            true_labels = add_one(list(range(len(tgt_paragraph))))  # 假设真实标签是顺序的 [1, 2, 3, 4, 5]
            best_score = float('-inf')
            best_predicted_labels = None
            # TODO: 生成所有可能的排列组合，并对每个排列进行评分，选择得分最高的排列作为最终结果
            for perm in itertools.permutations(range(len(tgt_paragraph))):
                predicted_labels = add_one(list(perm))
                predicted_paragraph = resort_paragraph(tgt_paragraph, predicted_labels)
                score = get_critic_score(model, predicted_paragraph)
                if score > best_score:
                    best_score = score
                    best_predicted_labels = predicted_labels
            # 3. 调用你的算分函数
            tau = cal_tau(best_predicted_labels, true_labels)
            acc = cal_acc(best_predicted_labels, true_labels)
            pmr = cal_PMR(best_predicted_labels, true_labels)
            
            if np.isnan(tau):
                tau = 0.0
            logger.warning(tau)
            total_tau += tau
            total_acc += acc
            total_pmr += pmr
            total_count += 1
            
        # 4. 计算当前模型的平均指标
        mean_tau = total_tau / total_count if total_count > 0 else 0
        mean_acc = total_acc / total_count if total_count > 0 else 0
        mean_pmr = total_pmr / total_count if total_count > 0 else 0
        logger.warning(f"Model: {file.name} -> Tau: {mean_tau:.4f}, Acc: {mean_acc:.4f}, PMR: {mean_pmr:.4f}")
        
        taus.append(mean_tau)
        accs.append(mean_acc)
        pmrs.append(mean_pmr)

        result_str = f"Tau: {mean_tau:.4f} | Acc: {mean_acc:.4f} | PMR: {mean_pmr:.4f}"
        
        # 5. 打印并记录日志
        print(f'Model: {file.name} -> {result_str}')
        common.logging.warning(f'Model: {file.name} -> {result_str}')

    # 6. 打印mean std
    print('=================== Mean & Std Across All Models ===================')
    print(f'tau: {common.cal_mean_std(taus)}')
    print(f'acc: {common.cal_mean_std(accs)}')
    print(f'pmr: {common.cal_mean_std(pmrs)}')
    logger.warning(f'Mean & Std Across All Models -> tau: {common.cal_mean_std(taus)}, acc: {common.cal_mean_std(accs)}, pmr: {common.cal_mean_std(pmrs)}')

    print("\n✅ 所有模型测试完毕！")