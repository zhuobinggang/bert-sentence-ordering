"""
测试只使用第一步解码的准确率。
这可以帮助我们判断multi_step_decode的性能差是因为错误累积还是因为第一步就有问题。
"""

from reader import *
import torch
from multi_step_decode import multi_step_decode

def first_step_decode(input_ids, attention_mask, bert=None):
    """
    只进行第一步解码：预测置信度最高的[MASK]。
    
    Args:
        input_ids: [512] token ids with MASK tokens
        attention_mask: [512] attention mask
        bert: model to use for decoding
        
    Returns:
        predicted_labels: list of first-step predicted labels (1-5), 但只有第一个元素有意义
    """
    toker = default_tokenizer()
    if bert is None:
        bert = default_bert()
    
    input_ids_tensor = torch.tensor([input_ids]).to(DEVICE)
    attention_mask_tensor = torch.tensor([attention_mask]).to(DEVICE)
    
    predicted_labels = []
    reversed_dict = reverse_indexs_tokenized()
    
    # 只进行一次预测
    with torch.no_grad():
        logits = bert(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor).logits  # [1, 512, 30522]
    
    # 找到所有MASK位置
    mask_positions = (input_ids_tensor[0] == toker.mask_token_id).nonzero(as_tuple=True)[0]
    
    if len(mask_positions) > 0:
        # 获取MASK位置的logits
        mask_logits = logits[0, mask_positions]  # [num_masks, 30522]
        
        # 对于每个MASK位置，获取最大logit值
        max_logits_per_position = mask_logits.max(dim=-1).values  # [num_masks]
        
        # 找到置信度最高的MASK位置
        best_mask_idx = max_logits_per_position.argmax(dim=-1).item()
        best_position = mask_positions[best_mask_idx].item()
        
        # 获取该位置的预测token id
        predicted_token_id = logits[0, best_position].argmax(dim=-1).item()
        
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
    
    Args:
        bert: model to use for decoding
        split: 'val' or 'test'
        num_samples: 如果指定，只验证前num_samples个样本；如果为None，则使用全部
        
    Returns:
        TestResult with tau, acc, pmr metrics (只用第一步的预测)
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
    for idx, bert_input in enumerate(tqdm(bert_inputs)):
        # 只进行第一步解码
        first_step_predictions = first_step_decode(bert_input.input_ids, bert_input.attention_mask, bert)
        
        if len(first_step_predictions) > 0:
            # 获取真实标签
            true_labels = [label for label in bert_input.labels if label != -100]
            true_labels = [reversed_dict[b] for b in true_labels]
            
            # 将第一步的预测结果复制到5个位置（虽然只有第一个是有意义的）
            # 这样可以进行准确率计算
            predicted_label = first_step_predictions[0]
            # 对于演示，我们只比较第一个位置的准确率
            all_predicted_labels.append([predicted_label])
            all_true_labels.append([true_labels[0]])  # 只比较第一个位置
        else:
            print(f"警告: 样本 {idx} 没有找到MASK位置")
    
    # 计算第一步的准确率
    if len(all_predicted_labels) > 0:
        correct_count = 0
        for pred_list, true_list in zip(all_predicted_labels, all_true_labels):
            if pred_list[0] == true_list[0]:
                correct_count += 1
        
        first_step_acc = correct_count / len(all_predicted_labels)
        print(f"\n=== First-Step Decoding Accuracy (only comparing position 1) ===")
        print(f"Correct predictions: {correct_count} / {len(all_predicted_labels)}")
        print(f"First-step accuracy: {first_step_acc:.4f}")
        print(f"================================================\n")
        
        return TestResult(tau=-1.0, acc=first_step_acc, pmr=-1.0)
    else:
        print("没有可用的样本进行测试")
        return TestResult(tau=-1.0, acc=0.0, pmr=-1.0)


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
    for bert_input in tqdm(bert_inputs):
        # 第一步解码
        first_step_predictions = first_step_decode(bert_input.input_ids, bert_input.attention_mask, bert)
        
        # 多步解码
        multi_step_predictions = multi_step_decode(bert_input.input_ids, bert_input.attention_mask, bert)
        
        # 真实标签
        true_labels = [label for label in bert_input.labels if label != -100]
        true_labels = [reversed_dict[b] for b in true_labels]
        
        assert len(multi_step_predictions) == len(true_labels) == 5, "应该有5个预测和真实标签"
        
        first_step_predictions_all.append(first_step_predictions)
        multi_step_predictions_all.append(multi_step_predictions)
        true_labels_all.append(true_labels)
    
    # 计算准确率
    print("\n=== Accuracy Comparison ===")
    
    # 第一步准确率（只比较第一个位置）
    first_step_correct = 0
    for pred_list, true_list in zip(first_step_predictions_all, true_labels_all):
        if len(pred_list) > 0 and pred_list[0] == true_list[0]:
            first_step_correct += 1
    first_step_acc = first_step_correct / len(first_step_predictions_all)
    print(f"First-step accuracy (position 1): {first_step_acc:.4f} ({first_step_correct}/{len(first_step_predictions_all)})")
    
    # 多步准确率 - 逐位置比较
    print("\nMulti-step accuracy by position:")
    for pos in range(5):
        correct = 0
        for pred_list, true_list in zip(multi_step_predictions_all, true_labels_all):
            if pos < len(pred_list) and pred_list[pos] == true_list[pos]:
                correct += 1
        acc = correct / len(multi_step_predictions_all)
        print(f"  Position {pos+1}: {acc:.4f} ({correct}/{len(multi_step_predictions_all)})")
    
    # 全序列准确率（所有5个位置都预测正确）
    full_seq_correct = 0
    for pred_list, true_list in zip(multi_step_predictions_all, true_labels_all):
        if pred_list == true_list:
            full_seq_correct += 1
    full_seq_acc = full_seq_correct / len(multi_step_predictions_all)
    print(f"\nFull sequence accuracy (all 5 positions): {full_seq_acc:.4f} ({full_seq_correct}/{len(multi_step_predictions_all)})")
    
    print("\n=== 分析 ===")
    print(f"第一步准确率: {first_step_acc:.4f}")
    print(f"多步解码整个序列准确率: {full_seq_acc:.4f}")
    print(f"准确率下降: {(first_step_acc - full_seq_acc):.4f}")
    print(f"下降百分比: {((first_step_acc - full_seq_acc) / first_step_acc * 100):.2f}%")
    print("(注: 如果第一步准确率很高但整个序列准确率很低，说明错误在累积)")


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
