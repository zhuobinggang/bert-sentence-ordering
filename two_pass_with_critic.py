from two_pass_plus import *
from critic_bert_simple import *
from critic_bert import resort_paragraph, recover_unsorted_paragraph

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

def run():
    bert1 = default_bert()
    load_checkpoint(bert1, './checkpoints/SIND_best_e1.pth' )
    bert1.to(DEVICE)
    bert1.eval()
    test_inputs = sind_data_prepare(sind_only_texts_get_by_split('test'))
    result = valid_bert_two_pass_plus(bert=bert1, bert_inputs=test_inputs)
    # 根据result.all_true_labels重建段落，然后根据all_predicted_labels_first和all_predicted_labels_second重建预测的段落，最后使用critic模型进行评分比较，看看是否能区分出first pass和second pass的结果差异
    paragraphs = sind_only_texts_get_by_split('test')
    original_order_paragraphs = []
    for paragraph, true_label in zip(paragraphs, result['all_true_labels']):
        original_order_paragraphs.append(recover_unsorted_paragraph(paragraph, true_label))
    critic = CriticBert()
    load_checkpoint(critic, './checkpoints/critic_bert_pointwise_good.pth')
    critic.to(DEVICE)
    critic.eval()
    all_true_labels = []
    all_preds = []
    printed = False
    for i in tqdm(range(len(original_order_paragraphs)), desc="Scoring with Critic"):
        paragraph = original_order_paragraphs[i]
        true_label = result['all_true_labels'][i]
        all_true_labels.append(true_label)
        pred_first = result['all_predicted_labels_first'][i]
        pred_second = result['all_predicted_labels_second'][i]
        if list_equal(pred_first, pred_second):
            # 如果两次预测结果相同，直接使用pred_first作为最终预测结果
            all_preds.append(pred_first)
        else:
            # 如果两次预测结果不同，使用critic模型进行评分比较
            para_first = resort_paragraph(paragraph, pred_first)
            score_first = get_critic_score(critic, para_first)
            para_second = resort_paragraph(paragraph, pred_second)
            score_second = get_critic_score(critic, para_second)
            if not printed:
                print(f'Original paragraph: {paragraphs[i]}')
                print(f"Resorted paragraph: {paragraph}")
                print(f"True label: {true_label}")
                print(f"Predicted first: {pred_first}, score: {score_first}")
                print(f"Predicted second: {pred_second}, score: {score_second}")
                printed = True
            if score_first > score_second:
                all_preds.append(pred_first)
            else:
                all_preds.append(pred_second)
    print("Final results after critic scoring:")
    final_result = cal_tau_acc_pmr(all_preds, all_true_labels, need_fix = False)
    print(final_result)
    return all_true_labels, all_preds
