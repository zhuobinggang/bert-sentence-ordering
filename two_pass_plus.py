# 考察two pass的详细信息
from two_pass_decode import *

def list_equal(list1, list2):
    if len(list1) != len(list2):
        return False
    for a, b in zip(list1, list2):
        if a != b:
            return False
    return True

def valid_bert_two_pass_plus(bert = None, split = 'val', bert_inputs = None):
    if bert is None:
        bert = default_bert()
    # 首先将val数据集转换成BertInput格式
    # paragraphs = sind_only_texts_get_by_split('val')[:100] # 取前100个故事进行测试
    if bert_inputs is None:
        paragraphs = sind_only_texts_get_by_split(split)
        bert_inputs = sind_data_prepare(paragraphs)
    # 然后使用默认的BERT模型进行解码，计算准确率和tau值
    all_predicted_labels_first = []
    all_predicted_labels_second = []
    all_true_labels = []
    reversed_dict = reverse_indexs_tokenized()
    for bert_input in tqdm(bert_inputs):
        # NOTE: 这里使用匈牙利算法解码，得到无重复的标签序列，且直接就是1-5的标签，不需要再转换了
        first_mask_token_5index_logits = get_mask_token_5index_logits(bert_input.input_ids, bert_input.attention_mask, bert) # (5,5)
        first_predicted_labels = hungarian_algorithm_best_order(first_mask_token_5index_logits.cpu().numpy())
        # 根据预测的标签重排true_labels
        true_labels = [label for label in bert_input.labels if label != -100]
        true_labels = [reversed_dict[b] for b in true_labels]
        second_mask_token_5index_logits = np.zeros((5, 5))
        # 重排input_ids并获取新的mask_token_5index_logits
        new_input_ids = resort_token_ids(bert_input.input_ids, first_predicted_labels)
        temp_mask_token_5index_logits = get_mask_token_5index_logits(new_input_ids, bert_input.attention_mask, bert) # 5,5
        for original_indice, target_indice in enumerate(first_predicted_labels):
            second_mask_token_5index_logits[original_indice] = temp_mask_token_5index_logits[target_indice - 1].cpu().numpy()
        second_predicted_labels = hungarian_algorithm_best_order(second_mask_token_5index_logits)
        # print(first_predicted_labels, second_predicted_labels)
        if not list_equal(second_predicted_labels, first_predicted_labels):
            if hasattr(valid_bert_two_pass_plus, "resorted_count"):
                valid_bert_two_pass_plus.resorted_count += 1
            else:
                valid_bert_two_pass_plus.resorted_count = 1
        # print_only_once(bert_input.input_ids, true_labels, first_predicted_labels, new_input_ids, 'XXX', second_predicted_labels)
        all_predicted_labels_first.append(first_predicted_labels)
        all_predicted_labels_second.append(second_predicted_labels)
        all_true_labels.append(true_labels)
    # 在修正标签之前计算一次
    print("First pass results:")
    test_result1 = cal_tau_acc_pmr(all_predicted_labels_first, all_true_labels, need_fix = False)
    print("Second pass results:")
    test_result2 = cal_tau_acc_pmr(all_predicted_labels_second, all_true_labels, need_fix = False)
    # print("Now fixing predicted labels and recalculating metrics...")
    # _ = cal_tau_acc(all_predicted_labels, all_true_labels, need_fix = True)
    return {
        'first_pass': test_result1,
        'second_pass': test_result2,
        'all_predicted_labels_first': all_predicted_labels_first,
        'all_predicted_labels_second': all_predicted_labels_second,
        'all_true_labels': all_true_labels
    }

# 首先对one pass和two pass进行精准的测试
def test_one_pass_vs_two_pass():
    bert = default_bert()
    load_checkpoint(bert, './checkpoints/SIND_best_e1.pth' )
    # valid_bert(bert, 'test')
    test_inputs = sind_data_prepare(sind_only_texts_get_by_split('test'))
    result = valid_bert_two_pass_plus(bert=bert, bert_inputs=test_inputs)
    print(f"Number of samples that were resorted: {getattr(valid_bert_two_pass_plus, 'resorted_count', 0)}")
    return result