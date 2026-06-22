# 使用匈牙利算法选出最优序列
from sind import *
import numpy as np
from scipy.optimize import linear_sum_assignment

# 使用匈牙利算法解码得到标签
def valid_bert_batched_decode(bert = None, split = 'val', split_length = None, dataloader = None):
    if bert is None:
        bert = default_bert()
    bert.eval()
    toker = default_tokenizer()
    # 首先将val数据集转换成BertInput格式
    # paragraphs = sind_only_texts_get_by_split('val')[:100] # 取前100个故事进行测试
    if dataloader is None:
        paragraphs = sind_paragraphs(split)
        if split_length is not None:
            common.print_once(f"只使用{split}前{split_length}个故事进行验证")
            paragraphs = paragraphs[:split_length]
        bert_inputs = sind_data_prepare(paragraphs)
        dataloader = bert_inputs_to_dataloader_shuffle(bert_inputs)
    # 然后使用默认的BERT模型进行解码，计算准确率和tau值
    all_predicted_labels = []
    all_true_labels = []
    reversed_dict = reverse_indexs_tokenized()
    index_dict = indexs_tokenized()
    index_1_to_5_token_ids = [index_dict[i] for i in range(1, 6)]
    for batch in tqdm(dataloader, desc="Validation"):
        input_ids, attention_mask, label_ids = batch
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        with torch.no_grad():
            logits = bert(input_ids=input_ids, attention_mask=attention_mask).logits # # [batch_size, 512, 30522]
        # mask_token_index = (input_ids == toker.mask_token_id).nonzero(as_tuple=True)
        for i in range(input_ids.size(0)): # 遍历batch中的每个样本
            mask_token_bool = (input_ids[i] == toker.mask_token_id)
            # predicted_token_ids = logits[i, mask_token_bool].argmax(axis=-1) # [5]
            predicted_token_ids = logits[i, mask_token_bool] # [5, vocab_size]
            predicted_token_ids = predicted_token_ids[:, index_1_to_5_token_ids] # [5, 5] 每个mask位置对应5个标签的logits
            predicted_labels = hungarian_algorithm_best_order(predicted_token_ids.cpu().numpy()) # [5] 每个位置的最终标签（1-5）
            true_label_ids = label_ids[i][label_ids[i] != -100] # [5]
            assert len(predicted_token_ids) == len(true_label_ids) == 5, "There should be exactly 5 predicted and true labels"
            # predicted_labels = [reversed_dict.get(a.item(), 5) for a in predicted_token_ids]
            true_labels = [reversed_dict[b.item()] for b in true_label_ids]
            all_predicted_labels.append(predicted_labels)
            all_true_labels.append(true_labels)
    # 在修正标签之前计算一次
    test_result = cal_tau_acc_pmr(all_predicted_labels, all_true_labels, need_fix = False)
    # print("Now fixing predicted labels and recalculating metrics...")
    # _ = cal_tau_acc_pmr(all_predicted_labels, all_true_labels, need_fix = True)
    return test_result

def hungarian_algorithm_best_order(model_outputs):
    """
    model_outputs: 模型的原始输出。
    假设形状为 (5, 5)，即 5个句子，每个句子对应 5个位置的 logit 或 softmax 概率值。
    model_outputs[i][j] 表示第 i 个句子填入位置 j (0-4) 的得分。
    """
    # 1. 转换为 numpy 数组
    score_matrix = np.array(model_outputs)
    
    # 2. 因为 scipy 的 linear_sum_assignment 寻找的是完美匹配的“最小代价”
    # 我们要找的是“最大得分”，所以将矩阵取负号，将其转化为求最小值问题
    cost_matrix = -score_matrix
    
    # 3. 运行匈牙利算法
    # row_ind 会是 [0, 1, 2, 3, 4] （句子的索引）
    # col_ind 会是算法分配的、绝对不重复的 [p0, p1, p2, p3, p4] （位置的索引）
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # 4. col_ind 就是最终生成的无重复完美排列（0~4 映射）
    # 如果你的评估代码需要 1~5 的标签，直接 + 1 即可
    final_positions = col_ind + 1
    
    return final_positions

def test_get_valid_permutation():
    predicted_token_ids = torch.rand(5, 9999)
    # reversed_dict = reverse_indexs_tokenized()
    index_dict = indexs_tokenized()
    index_1_to_5_token_ids = [index_dict[i] for i in range(1, 6)]
    dd = predicted_token_ids[:, index_1_to_5_token_ids] # [5, 5] 每个mask位置对应5个标签的logits
    return hungarian_algorithm_best_order(dd.cpu().numpy())

def test_valid_bert_batched():
    bert = default_bert()
    load_checkpoint(bert, './checkpoints/SIND_best_e1.pth' )
    # valid_bert(bert, 'test')
    valid_bert_batched_decode(bert, 'test')