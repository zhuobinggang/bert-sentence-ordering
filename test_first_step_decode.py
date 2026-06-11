"""
测试只使用第一步解码的准确率。
这可以帮助我们判断multi_step_decode的性能差是因为错误累积还是因为第一步就有问题。
"""

from reader import *
import torch
from multi_step_decode import multi_step_decode

def first_step_decode_all_masks(input_ids, attention_mask, bert=None):
    """
    第一步解码所有[MASK]位置。
    与multi_step_decode不同，这里不更新输入，而是在原始输入的基础上预测所有MASK位置。
    这样可以测试"没有错误累积"情况下的准确率。
    
    Args:
        input_ids: [512] token ids with MASK tokens
        attention_mask: [512] attention mask
        bert: model to use for decoding
        
    Returns:
        predicted_labels: list of predicted labels (1-5) for all MASK positions in order
    """
    toker = default_tokenizer()
    if bert is None:
        bert = default_bert()
    
    input_ids_tensor = torch.tensor([input_ids]).to(DEVICE)
    attention_mask_tensor = torch.tensor([attention_mask]).to(DEVICE)
    
    predicted_labels = []
    reversed_dict = reverse_indexs_tokenized()
    
    # 只进行一次前向传播，不更新输入
    with torch.no_grad():
        logits = bert(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor).logits  # [1, 512, 30522]
    
    # 找到所有MASK位置
    mask_positions = (input_ids_tensor[0] == toker.mask_token_id).nonzero(as_tuple=True)[0]
    
    # 对每个MASK位置进行预测
    for mask_pos in mask_positions:
        predicted_token_id = logits[0, mask_pos].argmax(dim=-1).item()
        
        # 将token id转换为标签(1-5)
        if predicted_token_id in reversed_dict:
            predicted_label = reversed_dict[predicted_token_id]
        else:
            decoded_token = toker.decode(predicted_token_id)
            print(f"警告: {predicted_token_id} = {decoded_token}, 不在编码字典中, 分配标签 5")
            predicted_label = 5
        predicted_labels.append(predicted_label)
    
    return predicted_labels


def valid_bert_first_step(bert=None, split='val', num_samples=None):
    """
    验证只使用第一步解码的准确率。
    第一步解码是指：在原始输入的基础上预测所有MASK位置，不更新输入。
    这样可以测试"没有错误累积"情况下的准确率。
    
    Args:
        bert: model to use for decoding
        split: 'val' or 'test'
        num_samples: 如果指定，只验证前num_samples个样本；如果为None，则使用全部
        
    Returns:
        TestResult with tau, acc, pmr metrics
    """
    if bert is None:
        bert = default_bert()
    reversed_dict = reverse_indexs_tokenized()
    
    paragraphs = sind_only_texts_get_by_split(split)
    if num_samples is not None:
        paragraphs = paragraphs[:num_samples]
    
    bert_inputs = sind_data_prepare(paragraphs)
    
    all_predicted_labels = []
    all_true_labels = []
    
    print(f"Testing first-step decoding on {len(bert_inputs)} samples from '{split}' split...")
    print("(第一步解码 = 不更新输入，对所有MASK位置只预测一次)\n")
    
    for bert_input in tqdm(bert_inputs):
        # 第一步解码：在原始输入上预测所有MASK位置，不更新
        predicted_labels = first_step_decode_all_masks(bert_input.input_ids, bert_input.attention_mask, bert)
        
        # 获取真实标签
        true_labels = [label for label in bert_input.labels if label != -100]
        true_labels = [reversed_dict[b] for b in true_labels]
        
        assert len(predicted_labels) == len(true_labels) == 5, "应该有5个预测和真实标签"
        
        all_predicted_labels.append(predicted_labels)
        all_true_labels.append(true_labels)
    
    # 计算准确率和指标
    test_result = cal_tau_acc_pmr(all_predicted_labels, all_true_labels, need_fix=False)
    return test_result


def compare_first_step_vs_multi_step(bert=None, split='val', num_samples=None):
    """
    比较第一步解码 vs 多步解码的性能。
    
    Args:
        bert: model to use for decoding
        split: 'val' or 'test'
        num_samples: 如果指定，只验证前num_samples个样本；如果为None，则使用全部
    """
    if bert is None:
        bert = default_bert()
    reversed_dict = reverse_indexs_tokenized()
    
    paragraphs = sind_only_texts_get_by_split(split)
    if num_samples is not None:
        paragraphs = paragraphs[:num_samples]
    
    bert_inputs = sind_data_prepare(paragraphs)
    
    first_step_predictions_all = []
    multi_step_predictions_all = []
    true_labels_all = []
    
    print(f"Comparing first-step vs multi-step decoding on {len(bert_inputs)} samples from '{split}' split...")
    print("(第一步解码 = 不更新输入，对所有MASK位置只预测一次)\n")
    
    for bert_input in tqdm(bert_inputs):
        # 第一步解码：在原始输入上预测所有MASK位置，不更新
        first_step_predictions = first_step_decode_all_masks(bert_input.input_ids, bert_input.attention_mask, bert)
        
        # 多步解码：逐步更新输入
        multi_step_predictions = multi_step_decode(bert_input.input_ids, bert_input.attention_mask, bert)
        
        # 真实标签
        true_labels = [label for label in bert_input.labels if label != -100]
        true_labels = [reversed_dict[b] for b in true_labels]
        
        assert len(first_step_predictions) == len(multi_step_predictions) == len(true_labels) == 5, "应该有5个预测和真实标签"
        
        first_step_predictions_all.append(first_step_predictions)
        multi_step_predictions_all.append(multi_step_predictions)
        true_labels_all.append(true_labels)
    
    # 计算准确率
    print("\n" + "=" * 70)
    print("=== Accuracy Comparison ===")
    print("=" * 70)
    
    # 第一步准确率 - 逐位置比较
    print("\nFirst-step decoding accuracy by position:")
    for pos in range(5):
        correct = 0
        for pred_list, true_list in zip(first_step_predictions_all, true_labels_all):
            if pred_list[pos] == true_list[pos]:
                correct += 1
        acc = correct / len(first_step_predictions_all)
        print(f"  Position {pos+1}: {acc:.4f} ({correct}/{len(first_step_predictions_all)})")
    
    # 第一步全序列准确率
    first_step_full_correct = 0
    for pred_list, true_list in zip(first_step_predictions_all, true_labels_all):
        if pred_list == true_list:
            first_step_full_correct += 1
    first_step_full_acc = first_step_full_correct / len(first_step_predictions_all)
    print(f"  Full sequence: {first_step_full_acc:.4f} ({first_step_full_correct}/{len(first_step_predictions_all)})")
    
    # 多步准确率 - 逐位置比较
    print("\nMulti-step decoding accuracy by position:")
    for pos in range(5):
        correct = 0
        for pred_list, true_list in zip(multi_step_predictions_all, true_labels_all):
            if pred_list[pos] == true_list[pos]:
                correct += 1
        acc = correct / len(multi_step_predictions_all)
        print(f"  Position {pos+1}: {acc:.4f} ({correct}/{len(multi_step_predictions_all)})")
    
    # 多步全序列准确率
    multi_step_full_correct = 0
    for pred_list, true_list in zip(multi_step_predictions_all, true_labels_all):
        if pred_list == true_list:
            multi_step_full_correct += 1
    multi_step_full_acc = multi_step_full_correct / len(multi_step_predictions_all)
    print(f"  Full sequence: {multi_step_full_acc:.4f} ({multi_step_full_correct}/{len(multi_step_predictions_all)})")
    
    # 分析
    print("\n" + "=" * 70)
    print("=== 分析 ===")
    print("=" * 70)
    print(f"第一步全序列准确率: {first_step_full_acc:.4f}")
    print(f"多步全序列准确率: {multi_step_full_acc:.4f}")
    print(f"准确率下降: {(first_step_full_acc - multi_step_full_acc):.4f}")
    
    if first_step_full_acc > 0:
        drop_pct = (first_step_full_acc - multi_step_full_acc) / first_step_full_acc * 100
        print(f"下降百分比: {drop_pct:.2f}%")
    
    if first_step_full_acc > multi_step_full_acc:
        print("\n✓ 第一步性能更好，说明多步解码中存在错误累积问题")
    elif first_step_full_acc < multi_step_full_acc:
        print("\n✓ 多步性能更好，说明多步解码的更新策略有帮助")
    else:
        print("\n✓ 两者性能相同")
    
    print("=" * 70)


# Example usage
if __name__ == '__main__':
    bert = default_bert()
    load_checkpoint(bert, './checkpoints/SIND_best_e1.pth')
    bert.eval()
    
    print("=" * 60)
    print("Testing First-Step Decoding Accuracy")
    print("=" * 60)
    
    # 测试第一步准确率（使用前100个样本）
    result = valid_bert_first_step(bert, 'val', num_samples=100)
    
    print("\n" + "=" * 60)
    print("Comparing First-Step vs Multi-Step Decoding")
    print("=" * 60)
    
    # 比较第一步 vs 多步（使用前100个样本）
    compare_first_step_vs_multi_step(bert, 'val', num_samples=100)
