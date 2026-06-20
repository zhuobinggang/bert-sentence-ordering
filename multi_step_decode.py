# 多步解码逻辑：
# 每次预测之后选择置信度最高的一个位置进行输出，填充到输入中，继续下一轮预测，直到所有位置都被预测出来。

from sind import *
import torch
from common import args

def decode_by_bert_one_step(input_ids, attention_mask, bert=None):
    toker = default_tokenizer()
    if bert is None:
        bert = default_bert()
    input_ids = torch.tensor([input_ids]).to(DEVICE)
    attention_mask = torch.tensor([attention_mask]).to(DEVICE)
    with torch.no_grad():
        logits = bert(input_ids = input_ids, attention_mask = attention_mask).logits # [1, 512, 30522]
    mask_token_bool = (input_ids == toker.mask_token_id)[0] # [512]的bool张量，True表示对应位置是[mask] token
    mask_token_indices = torch.where(mask_token_bool)[0]  # 获取所有mask token的索引位置
    predicted_token_ids = logits[0, mask_token_bool].argmax(axis=-1)  # [5]
    # 只保留logits最大的predicted_token_ids和对应的true_label_ids
    max_logits = logits[0, mask_token_bool].max(dim=-1).values # 预测的最大logits值
    max_logits_index = max_logits.argmax() # 最大logits值的索引
    predicted_token_id = predicted_token_ids[max_logits_index].item() # 只保留最大logits对应的预测标签
    decoded_indice = mask_token_indices[max_logits_index].item() # 获取对应的输入位置索引
    return predicted_token_id, decoded_indice

def valid_bert_n_steps(bert = None, split = 'val', num_samples=None, n_steps=2):
    if bert is None:
        bert = default_bert()
    # 首先将val数据集转换成BertInput格式
    # paragraphs = sind_only_texts_get_by_split('val')[:100] # 取前100个故事进行测试
    paragraphs = sind_only_texts_get_by_split(split)
    if num_samples is not None:
        paragraphs = paragraphs[:num_samples]
    bert_inputs = sind_data_prepare(paragraphs)
    # 然后使用默认的BERT模型进行解码，计算准确率和tau值
    all_predicted_labels = []
    all_true_labels = []
    reversed_dict = reverse_indexs_tokenized()
    for bert_input in tqdm(bert_inputs):
        step_predicted_labels = []
        step_true_labels = []
        for step in range(n_steps):
            predicted_token_id, decoded_indice = decode_by_bert_one_step(bert_input.input_ids, bert_input.attention_mask, bert) # 注意要传递attention_mask
            # true_label_ids = [label for label in bert_input.labels if label != -100]
            true_label_id = bert_input.labels[decoded_indice]
            predicted_label = reversed_dict.get(predicted_token_id, 5) # 如果预测的token_id不在字典中，默认标签为5（表示无法预测）
            true_label = reversed_dict[true_label_id]
            step_predicted_labels.append(predicted_label)
            step_true_labels.append(true_label)
            bert_input.input_ids[decoded_indice] = predicted_token_id # 将预测的token_id替换到输入中，进行下一步解码
        all_predicted_labels.append(step_predicted_labels)
        all_true_labels.append(step_true_labels)
    # assert len(all_predicted_labels) == n_steps * len(bert_inputs), "预测标签数量应该是输入数量的两倍"
    # test_result = cal_tau_acc_pmr(all_predicted_labels, all_true_labels, need_fix = False)
    return all_predicted_labels, all_true_labels

def valid_bert_n_steps_flatten_acc(bert = None, split = 'val', num_samples=100, n_steps=2):
    all_predicted_labels, all_true_labels = valid_bert_n_steps(bert, split, num_samples, n_steps)
    all_predicted_labels = [a for nest_list in all_predicted_labels for a in nest_list] # 过滤掉无法预测的标签
    all_true_labels = [a for nest_list in all_true_labels for a in nest_list]
    print(cal_acc(all_predicted_labels, all_true_labels))

def valid_bert_5_steps(bert = None, split = 'val', num_samples=None):
    all_predicted_labels, all_true_labels = valid_bert_n_steps(bert, split, num_samples, n_steps=5)
    all_predicted_labels = [a for nest_list in all_predicted_labels for a in nest_list] # 过滤掉无法预测的标签
    all_true_labels = [a for nest_list in all_true_labels for a in nest_list]
    print(cal_tau_acc_pmr(all_predicted_labels, all_true_labels, need_fix = False))


# Example usage
# 1-step decoding accuracy: 0.88
# 2-step decoding accuracy: 0.7150
# 3-step decoding accuracy: 0.6500
# 4-step decoding accuracy: 0.5925
# 5-step decoding accuracy: 0.5480
if __name__ == '__main__':
    bert = default_bert()
    load_checkpoint(bert, './checkpoints/SIND_best_e1.pth')
    bert.eval()
    
    print("=" * 60)
    print("Testing First-Step Decoding Accuracy")
    print("=" * 60)
    
    # 测试第一步准确率
    # valid_bert_first_step(bert, 'val', num_samples=100)
    all_predicted_labels, all_true_labels = valid_bert_n_steps(bert, 'test', n_steps=5)
    cal_tau_acc_pmr(all_predicted_labels, all_true_labels, need_fix=False)

