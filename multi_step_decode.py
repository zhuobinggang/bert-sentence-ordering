# 多步解码逻辑：
# 每次预测之后选择置信度最高的一个位置进行输出，填充到输入中，继续下一轮预测，直到所有位置都被预测出来。

from reader import *
import torch

def multi_step_decode(input_ids, attention_mask, bert=None):
    """
    Multi-step decoding: iteratively predict one position with highest confidence at each step.
    
    Args:
        input_ids: [512] token ids with MASK tokens
        attention_mask: [512] attention mask
        bert: model to use for decoding
        
    Returns:
        predicted_labels: list of predicted labels (1-5) in the order of MASK positions
    """
    toker = default_tokenizer()
    if bert is None:
        bert = default_bert()
    
    # Convert to tensors (need to be mutable for the loop)
    input_ids_tensor = torch.tensor([input_ids]).to(DEVICE)
    attention_mask_tensor = torch.tensor([attention_mask]).to(DEVICE)
    
    predicted_labels = []
    reversed_dict = reverse_indexs_tokenized()
    
    # Iteratively predict one position at a time
    while True:
        # Find current MASK positions
        mask_positions = (input_ids_tensor[0] == toker.mask_token_id).nonzero(as_tuple=True)[0]
        
        if len(mask_positions) == 0:
            break
        
        # Get model predictions for all MASK positions
        with torch.no_grad():
            logits = bert(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor).logits  # [1, 512, 30522]
        
        # Get logits for MASK positions
        mask_logits = logits[0, mask_positions]  # [num_masks, 30522]
        
        # For each MASK position, get max logit value across vocabulary
        # max_logits[i] = max logit value for i-th MASK position
        max_logits_per_position = mask_logits.max(dim=-1).values  # [num_masks]
        
        # Find the position with highest confidence (max logit)
        best_mask_idx = max_logits_per_position.argmax(dim=-1).item()
        best_position = mask_positions[best_mask_idx].item()
        
        # Get the predicted token id for this position
        predicted_token_id = logits[0, best_position].argmax(dim=-1).item()
        
        # Convert token id to label (1-5)
        if predicted_token_id in reversed_dict:
            predicted_label = reversed_dict[predicted_token_id]
        else:
            decoded_token = toker.decode(predicted_token_id)
            print(f"警告: {predicted_token_id} = {decoded_token}, 不在编码字典中, 分配标签 5")
            predicted_label = 5
        predicted_labels.append(predicted_label)
        
        # Fill the predicted token at this position in the input
        input_ids_tensor[0, best_position] = predicted_token_id
    
    return predicted_labels


def valid_bert_multi_step(bert=None, split='val'):
    """
    Validation using multi-step decoding.
    
    Args:
        bert: model to use for decoding
        split: 'val' or 'test'
        
    Returns:
        TestResult with tau, acc, pmr metrics
    """
    if bert is None:
        bert = default_bert()
    reversed_dict = reverse_indexs_tokenized()
    
    paragraphs = sind_only_texts_get_by_split(split)[:10]
    bert_inputs = sind_data_prepare(paragraphs)
    
    all_predicted_labels = []
    all_true_labels = []
    
    for bert_input in tqdm(bert_inputs):
        predicted_labels = multi_step_decode(bert_input.input_ids, bert_input.attention_mask, bert)
        true_labels = [label for label in bert_input.labels if label != -100]
        true_labels = [reversed_dict[b] for b in true_labels]
        
        assert len(predicted_labels) == len(true_labels) == 5, "There should be exactly 5 predicted and true labels"
        
        all_predicted_labels.append(predicted_labels)
        all_true_labels.append(true_labels)
    
    test_result = cal_tau_acc_pmr(all_predicted_labels, all_true_labels, need_fix=False)
    # print('all predicted labels:\n', all_predicted_labels)
    # print('all true labels:\n', all_true_labels)
    return test_result


# Example usage
if __name__ == '__main__':
    bert = default_bert()
    load_checkpoint(bert, './checkpoints/SIND_best_e1.pth')
    result = valid_bert_multi_step(bert, 'val')
