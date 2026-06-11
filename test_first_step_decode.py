"""
测试只使用第一步解码的准确率。
参考reader.py中的valid_bert_batched函数，简化测试逻辑。
"""

from reader import *
import torch


# First-step full sequence accuracy: 0.8800
def valid_bert_first_step(bert=None, split='val', num_samples=None):
    """
    验证第一步解码的准确率（不更新输入，只在原始输入上预测一次）。
    参考valid_bert_batched的逻辑。
    
    Args:
        bert: model to use for decoding
        split: 'val' or 'test'
        num_samples: 如果指定，只验证前num_samples个样本；如果为None，则使用全部
        
    Returns:
        准确率 (float)
    """
    if bert is None:
        bert = default_bert()
    
    toker = default_tokenizer()
    paragraphs = sind_only_texts_get_by_split(split)
    if num_samples is not None:
        paragraphs = paragraphs[:num_samples]
    
    bert_inputs = sind_data_prepare(paragraphs)
    dataloader = bert_inputs_to_dataloader(bert_inputs)
    
    all_predicted_labels = []
    all_true_labels = []
    reversed_dict = reverse_indexs_tokenized()
    
    print(f"Testing first-step decoding on {split} split (不更新输入，只预测一次)...")
    for batch in tqdm(dataloader, desc="First-step validation"):
        input_ids, attention_mask, label_ids = batch
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        
        with torch.no_grad():
            logits = bert(input_ids=input_ids, attention_mask=attention_mask).logits  # [batch_size, 512, 30522]
        
        for i in range(input_ids.size(0)):
            mask_token_bool = (input_ids[i] == toker.mask_token_id)
            predicted_token_ids = logits[i, mask_token_bool].argmax(axis=-1)  # [5]
            true_label_ids = label_ids[i][label_ids[i] != -100]  # [5]

            # 只保留logits最大的predicted_token_ids和对应的true_label_ids
            max_logits = logits[i, mask_token_bool].max(dim=-1).values # 预测的最大logits值
            max_logits_index = max_logits.argmax() # 最大logits值的索引
            predicted_token_ids = predicted_token_ids[max_logits_index].unsqueeze(0) # 只保留最大logits对应的预测标签
            true_label_ids = true_label_ids[max_logits_index].unsqueeze(0) # 只保留最大logits对应的真实


            assert len(predicted_token_ids) == len(true_label_ids) == 1
            
            predicted_labels = [reversed_dict.get(a.item(), 5) for a in predicted_token_ids]
            true_labels = [reversed_dict[b.item()] for b in true_label_ids]
            
            all_predicted_labels.append(predicted_labels)
            all_true_labels.append(true_labels)
    
    # 计算准确率
    accs = []
    for predicted_labels, true_labels in zip(all_predicted_labels, all_true_labels):
        acc = cal_acc(predicted_labels, true_labels)
        accs.append(acc)
    avg_acc = sum(accs) / len(accs)
    
    print(f"\nFirst-step full sequence accuracy: {avg_acc:.4f}")
    
    return avg_acc


def decode_by_bert_one_step(input_ids, attention_mask, bert=None):
    toker = default_tokenizer()
    if bert is None:
        bert = default_bert()
    input_ids = torch.tensor([input_ids]).to(DEVICE)
    attention_mask = torch.tensor([attention_mask]).to(DEVICE)
    with torch.no_grad():
        logits = bert(input_ids = input_ids, attention_mask = attention_mask).logits # [1, 512, 30522]
    mask_token_bool = (input_ids == toker.mask_token_id)
    predicted_token_ids = logits[mask_token_bool].argmax(axis=-1)  # [5]
    assert len(predicted_token_ids) == 5, "There should be exactly 5 predicted token ids"
    return predicted_token_ids.tolist()

def valid_bert_two_steps(bert = None, split = 'val', num_samples=100):
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
        predicted_labels = decode_by_bert_one_step(bert_input.input_ids, bert_input.attention_mask, bert) # 注意要传递attention_mask
        true_labels = [label for label in bert_input.labels if label != -100]
        assert len(predicted_labels) == len(true_labels) == 5, "There should be exactly 5 predicted and true labels"
        predicted_labels = [reversed_dict.get(a, 5) for a in predicted_labels]
        true_labels = [reversed_dict[b] for b in true_labels]
        all_predicted_labels.append(predicted_labels)
        all_true_labels.append(true_labels)
    # 在修正标签之前计算一次
    test_result = cal_tau_acc_pmr(all_predicted_labels, all_true_labels, need_fix = False)
    # print("Now fixing predicted labels and recalculating metrics...")
    # _ = cal_tau_acc(all_predicted_labels, all_true_labels, need_fix = True)
    return test_result

# Example usage
if __name__ == '__main__':
    bert = default_bert()
    load_checkpoint(bert, './checkpoints/SIND_best_e1.pth')
    bert.eval()
    
    print("=" * 60)
    print("Testing First-Step Decoding Accuracy")
    print("=" * 60)
    
    # 测试第一步准确率
    # valid_bert_first_step(bert, 'val', num_samples=100)
    valid_bert_two_steps(bert, 'val', num_samples=100)
