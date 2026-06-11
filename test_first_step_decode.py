"""
测试只使用第一步解码的准确率。
参考reader.py中的valid_bert_batched函数，简化测试逻辑。
"""

from reader import *
import torch

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
            
            assert len(predicted_token_ids) == len(true_label_ids) == 5
            
            predicted_labels = [reversed_dict.get(a.item(), 5) for a in predicted_token_ids]
            true_labels = [reversed_dict[b.item()] for b in true_label_ids]
            
            all_predicted_labels.append(predicted_labels)
            all_true_labels.append(true_labels)
    
    # 计算全序列准确率
    correct = sum(1 for p, t in zip(all_predicted_labels, all_true_labels) if p == t)
    accuracy = correct / len(all_predicted_labels)
    
    print(f"\nFirst-step full sequence accuracy: {accuracy:.4f} ({correct}/{len(all_predicted_labels)})")
    
    return accuracy


# Example usage
if __name__ == '__main__':
    bert = default_bert()
    load_checkpoint(bert, './checkpoints/SIND_best_e1.pth')
    bert.eval()
    
    print("=" * 60)
    print("Testing First-Step Decoding Accuracy")
    print("=" * 60)
    
    # 测试第一步准确率
    valid_bert_first_step(bert, 'val', num_samples=100)
