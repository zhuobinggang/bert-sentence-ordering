# 解码一次之后重新排序，再解码一次，看看能不能提升性能
from sind import *
import numpy as np


def print_only_once(input_ids, labels, predicted_labels, new_input_ids, new_labels, new_predicted_labels):
    if not hasattr(print_only_once, "has_printed"):
        input_ids = [x for x in input_ids if x != default_tokenizer().pad_token_id]
        print(f'Original input_ids: {default_tokenizer().decode(input_ids)}')
        print(f'Original predicted labels: {predicted_labels}')
        print(f'Original true labels: {labels}')
        new_input_ids = [x for x in new_input_ids if x != default_tokenizer().pad_token_id]
        print(f'New input_ids after resorting: {default_tokenizer().decode(new_input_ids)}')
        print(f'New true labels after resorting: {new_labels}')
        print(f'New predicted labels after second pass: {new_predicted_labels}')
        print_only_once.has_printed = True

def valid_bert_two_pass(bert = None, split = 'val', bert_inputs = None):
    if bert is None:
        bert = default_bert()
    # 首先将val数据集转换成BertInput格式
    # paragraphs = sind_only_texts_get_by_split('val')[:100] # 取前100个故事进行测试
    if bert_inputs is None:
        paragraphs = sind_paragraphs(split)
        bert_inputs = sind_data_prepare(paragraphs)
    # 然后使用默认的BERT模型进行解码，计算准确率和tau值
    all_predicted_labels = []
    all_true_labels = []
    reversed_dict = reverse_indexs_tokenized()
    for bert_input in tqdm(bert_inputs):
        # NOTE: 这里使用匈牙利算法解码，得到无重复的标签序列，且直接就是1-5的标签，不需要再转换了
        predicted_labels = decode_by_bert(bert_input.input_ids, bert_input.attention_mask, bert) 
        true_labels = [label for label in bert_input.labels if label != -100]
        true_labels = [reversed_dict[b] for b in true_labels]
        # 根据预测的标签重排true_labels
        new_true_labels = [0] * 5
        for i, label in enumerate(predicted_labels):
            new_true_labels[label - 1] = true_labels[i]
        # 二次解码
        # 重排input_ids
        new_input_ids = resort_token_ids(bert_input.input_ids, predicted_labels)
        new_predicted_labels = decode_by_bert(new_input_ids, bert_input.attention_mask, bert) 
        if new_predicted_labels != predicted_labels:
            if hasattr(valid_bert_two_pass, "resorted_count"):
                valid_bert_two_pass.resorted_count += 1
            else:
                valid_bert_two_pass.resorted_count = 1
        print_only_once(bert_input.input_ids, true_labels, predicted_labels, new_input_ids, new_true_labels, new_predicted_labels)
        all_predicted_labels.append(new_predicted_labels)
        all_true_labels.append(new_true_labels)
    # 在修正标签之前计算一次
    test_result = cal_tau_acc_pmr(all_predicted_labels, all_true_labels, need_fix = False)
    # print("Now fixing predicted labels and recalculating metrics...")
    # _ = cal_tau_acc(all_predicted_labels, all_true_labels, need_fix = True)
    return test_result


def input_ids_to_slices(input_ids):
    toker = default_tokenizer()
    mask_token_indices = [i for i, token_id in enumerate(input_ids) if token_id == toker.mask_token_id]
    slices = []
    slice = []
    for i, idx in enumerate(input_ids):
        if idx == toker.sep_token_id:
            break
        if i + 1 in mask_token_indices:
            slices.append(slice)
            slice = []
        slice.append(idx)
    slices.append(slice)
    slices.append([toker.sep_token_id])
    return slices


# [CLS]
# ( [MASK] ) sentence five.
# ( [MASK] ) sentence one.
# ( [MASK] ) sentence two.
# ( [MASK] ) sentence three.
# ( [MASK] ) sentence four.
# [SEP]
def test_input_ids_to_slices():
    paragraph = ["Sentence one.", "Sentence two.", "Sentence three.", "Sentence four.", "Sentence five."]
    bert_input = sind_data_prepare([paragraph])[0]
    ss = input_ids_to_slices(bert_input.input_ids)
    for s in ss:
        print(default_tokenizer().decode(s))


def resort_token_ids(input_ids, predicted_labels_org):
    predicted_labels = predicted_labels_org.copy()
    assert min(predicted_labels) >= 1 and max(predicted_labels) <= 5, "Predicted labels should be in the range of 1 to 5"
    slices = input_ids_to_slices(input_ids)
    slice_cls = slices[0]
    slice_sep = slices[-1]
    sentence_slices = slices[1:-1]
    # 根据predicted_labels重新排序sentence_slices
    predicted_labels = [label - 1 for label in predicted_labels] # 转换为0-4的索引
    sorted_sentence_slices = [None] * 5
    for i, label in enumerate(predicted_labels):
        sorted_sentence_slices[label] = sentence_slices[i]
    # 将重新排序后的切片拼接成新的input_ids
    new_input_ids = slice_cls
    for s in sorted_sentence_slices:
        new_input_ids += s
    new_input_ids += slice_sep
    # 如果需要重新pad的话
    toker = default_tokenizer()
    if toker.pad_token_id in input_ids:
        new_input_ids += [toker.pad_token_id] * (len(input_ids) - len(new_input_ids)) # padding
    return new_input_ids

# Tested 2026.6.15
def test_resort_token_ids():
    paragraph = ["Sentence one.", "Sentence two.", "Sentence three.", "Sentence four.", "Sentence five."]
    bert_input = sind_data_prepare([paragraph])[0]
    print(default_tokenizer().decode(bert_input.input_ids))
    predicted_labels = [5, 4, 3, 2, 1] # 假设模型预测的标签是这样的
    new_input_ids = resort_token_ids(bert_input.input_ids, predicted_labels)
    print(default_tokenizer().decode(new_input_ids))


def test_valid_bert_two_pass():
    bert = default_bert()
    load_checkpoint(bert, './checkpoints/SIND_best_e1.pth' )
    # valid_bert(bert, 'test')
    valid_bert_two_pass(bert, 'test')
    print(f"Number of samples that were resorted: {getattr(valid_bert_two_pass, 'resorted_count', 0)}")