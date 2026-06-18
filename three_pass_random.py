# 打乱输入顺序三次，句子输出概率叠加
from two_pass_decode import *
import random

def valid_bert_three_pass_random(bert = None, split = 'val', bert_inputs = None):
    if bert is None:
        bert = default_bert()
    # 首先将val数据集转换成BertInput格式
    # paragraphs = sind_only_texts_get_by_split('val')[:100] # 取前100个故事进行测试
    if bert_inputs is None:
        paragraphs = sind_only_texts_get_by_split(split)
        bert_inputs = sind_data_prepare(paragraphs)
    # 然后使用默认的BERT模型进行解码，计算准确率和tau值
    all_predicted_labels = []
    all_true_labels = []
    reversed_dict = reverse_indexs_tokenized()
    for bert_input in tqdm(bert_inputs):
        # NOTE: 这里使用匈牙利算法解码，得到无重复的标签序列，且直接就是1-5的标签，不需要再转换了
        mask_token_5index_logits = get_mask_token_5index_logits(bert_input.input_ids, bert_input.attention_mask, bert) # (5,5)
        # predicted_labels = hungarian_algorithm_best_order(mask_token_logits.cpu().numpy())
        true_labels = [label for label in bert_input.labels if label != -100]
        true_labels = [reversed_dict[b] for b in true_labels]
        # 加两次解码
        for _ in range(2):
            random_indices  = add_one(random.sample(range(5), 5))
            new_input_ids = resort_token_ids(bert_input.input_ids, random_indices)
            temp_mask_token_5index_logits = get_mask_token_5index_logits(new_input_ids, bert_input.attention_mask, bert)
            for original_indice, target_indice in enumerate(random_indices):
                mask_token_5index_logits[original_indice] += temp_mask_token_5index_logits[target_indice - 1]
        mask_token_5index_logits = mask_token_5index_logits / 2
        predicted_labels = hungarian_algorithm_best_order(mask_token_5index_logits.cpu().numpy())
        all_predicted_labels.append(predicted_labels)
        all_true_labels.append(true_labels)
    # 在修正标签之前计算一次
    test_result = cal_tau_acc_pmr(all_predicted_labels, all_true_labels, need_fix = False)
    # print("Now fixing predicted labels and recalculating metrics...")
    # _ = cal_tau_acc(all_predicted_labels, all_true_labels, need_fix = True)
    return test_result

def test_valid_bert_three_pass_random():
    bert = default_bert()
    load_checkpoint(bert, './checkpoints/SIND_best_e1.pth' )
    # valid_bert(bert, 'test')
    valid_bert_three_pass_random(bert, 'test')