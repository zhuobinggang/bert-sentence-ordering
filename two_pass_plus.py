# 考察two pass的详细信息
from two_pass_decode import *
from common import list_equal


def valid_bert_two_pass_plus(bert = None, split = 'val', bert_inputs = None):
    if bert is None:
        bert = default_bert()
    # 首先将val数据集转换成BertInput格式
    # paragraphs = sind_only_texts_get_by_split('val')[:100] # 取前100个故事进行测试
    if bert_inputs is None:
        paragraphs = sind_paragraphs(split)
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


def valid_bert_two_pass_average(bert = None, split = 'val', bert_inputs = None):
    if bert is None:
        bert = default_bert()
    # 首先将val数据集转换成BertInput格式
    # paragraphs = sind_only_texts_get_by_split('val')[:100] # 取前100个故事进行测试
    if bert_inputs is None:
        paragraphs = sind_paragraphs(split)
        bert_inputs = sind_data_prepare(paragraphs)
    # 然后使用默认的BERT模型进行解码，计算准确率和tau值
    all_predicted_labels_first = []
    all_predicted_labels_second = []
    all_true_labels = []
    reversed_dict = reverse_indexs_tokenized()
    for bert_input in tqdm(bert_inputs):
        # NOTE: 这里使用匈牙利算法解码，得到无重复的标签序列，且直接就是1-5的标签，不需要再转换了
        first_mask_token_5index_logits = get_mask_token_5index_logits(bert_input.input_ids, bert_input.attention_mask, bert) # (5,5)
        first_mask_token_5index_logits = first_mask_token_5index_logits.cpu().numpy()
        first_predicted_labels = hungarian_algorithm_best_order(first_mask_token_5index_logits)
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
            if hasattr(valid_bert_two_pass_average, "resorted_count"):
                valid_bert_two_pass_average.resorted_count += 1
            else:
                valid_bert_two_pass_average.resorted_count = 1
            first_mask_token_5index_logits = (first_mask_token_5index_logits + second_mask_token_5index_logits) / 2
            first_predicted_labels = hungarian_algorithm_best_order(first_mask_token_5index_logits)
        all_predicted_labels_first.append(first_predicted_labels)
        all_true_labels.append(true_labels)
    test_result = cal_tau_acc_pmr(all_predicted_labels_first, all_true_labels, need_fix = False)
    return test_result


def cal_confident_results(all_predicted_labels_first, all_predicted_labels_second, all_true_labels):
    confident_samples = []
    for first, second, true in zip(all_predicted_labels_first, all_predicted_labels_second, all_true_labels):
        if list_equal(first, second):
            confident_samples.append((first, true))
    print(f"Number of confident samples: {len(confident_samples)}")
    print(f'confident samples ratio: {len(confident_samples) / len(all_true_labels):.4f}')
    if len(confident_samples) > 0:
        confident_predicted = [sample[0] for sample in confident_samples]
        confident_true = [sample[1] for sample in confident_samples]
        test_result_confident = cal_tau_acc_pmr(confident_predicted, confident_true, need_fix = False)
        print(f"Confident samples test result: {test_result_confident}")

# 在不确信的样本中，分析如果最好的情况下总能选取更好的结果，那么性能提升的上限是多少
def cal_potential_improvement(all_predicted_labels_first, all_predicted_labels_second, all_true_labels, degrade = False):
    all_taus = []
    for first, second, true in zip(all_predicted_labels_first, all_predicted_labels_second, all_true_labels):
        score = cal_tau(first, true)
        if not list_equal(first, second):
            second_score = cal_tau(second, true)
            if degrade:
                if second_score < score:
                    best_score = second_score
            if second_score > score:
                best_score = second_score
        all_taus.append(best_score)
    average_best_tau = sum(all_taus) / len(all_taus)
    print(f"Tau in perfect scenario: {average_best_tau:.4f}")

# 首先对one pass和two pass进行精准的测试
def test_two_pass():
    bert = default_bert()
    load_checkpoint(bert, './checkpoints/SIND_best_e1.pth' )
    # valid_bert(bert, 'test')
    test_inputs = sind_data_prepare(sind_paragraphs('test'))
    result = valid_bert_two_pass_plus(bert=bert, bert_inputs=test_inputs)
    # print(f"Number of samples that were resorted: {getattr(valid_bert_two_pass_plus, 'resorted_count', 0)}")
    cal_confident_results(result['all_predicted_labels_first'], result['all_predicted_labels_second'], result['all_true_labels'])
    print('现在计算平均后的性能')
    result_average = valid_bert_two_pass_average(bert=bert, bert_inputs=test_inputs)



