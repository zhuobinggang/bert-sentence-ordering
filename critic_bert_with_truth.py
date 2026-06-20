# 用bert来判断两个段落哪个更好
from two_pass_plus import *
import torch
from torch import nn
from tqdm import tqdm
from sind import save_checkpoint, load_checkpoint
import json

MAX_SENTENCE_TOKENS = 50 # 因为有10句话，BERT的最大输入限制是512，所以平均每句话不能超过50个token

def dataset_create(split = 'train'):
    bert = default_bert()
    load_checkpoint(bert, './checkpoints/SIND_best_e1.pth' )
    # valid_bert(bert, 'test')
    paragraphs = sind_paragraphs(split)
    train_inputs = sind_data_prepare(paragraphs)
    result = valid_bert_two_pass_plus(bert=bert, bert_inputs=train_inputs) # 这个要10分钟
    # 按照result['all_true_labels']的顺序对段落进行排序
    original_order_paragraphs = []
    for paragraph, true_label in zip(paragraphs, result['all_true_labels']):
        ordered_paragraph = [None] * 5
        for idx, label in enumerate(true_label):
            ordered_paragraph[idx] = paragraph[label - 1] # label是1-5的索引
        original_order_paragraphs.append(ordered_paragraph)
    # 将original_order_paragraphs和对应的all_true_labels, all_predicted_labels_first, all_predicted_labels_second一起保存到文件中
    output_data = []
    for paragraph, true_label, pred_first, pred_second in zip(original_order_paragraphs, result['all_true_labels'], result['all_predicted_labels_first'], result['all_predicted_labels_second']):
        if hasattr(pred_first[0], 'tolist'):
            pred_first = [label.tolist() for label in pred_first]
        if hasattr(pred_second[0], 'tolist'):
            pred_second = [label.tolist() for label in pred_second]
        output_data.append({
            'paragraph': paragraph,
            'true_label': true_label,
            'predicted_label_first': pred_first,
            'predicted_label_second': pred_second
        })
    with open(f'./temp_datasets/{split}_two_pass_results.json', 'w') as f:
        json.dump(output_data, f, indent=4)

class CriticBert(nn.Module):
    def __init__(self):
        super(CriticBert, self).__init__()
        self.bert = default_bert()
        self.head = nn.Linear(self.bert.config.hidden_size, 1) # 去掉 nn.Sigmoid()

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True) 
        last_hidden_state = outputs.hidden_states[-1] 
        cls_hidden_state = last_hidden_state[:, 0, :] 
        logits = self.head(cls_hidden_state) # 输出的是实数 Logits，而不是 0~1 的概率
        return logits

def bert_input_critic_bert(paragraph, predicted_label_first, predicted_label_second):
    # 将段落按照predicted_label_first的顺序进行排序
    ordered_paragraph_first = [None] * 5
    for idx, label in enumerate(predicted_label_first):
        ordered_paragraph_first[label - 1] = paragraph[idx]
    # 将段落按照predicted_label_second的顺序进行排序
    ordered_paragraph_second = [None] * 5
    for idx, label in enumerate(predicted_label_second):
        ordered_paragraph_second[label - 1] = paragraph[idx]
    # 将ordered_paragraph_first和ordered_paragraph_second拼接送入BERT，输出first更好的概率
    token_type_ids = []
    inputs_ids = []
    inputs_ids.append(default_tokenizer().cls_token_id)
    for sent in ordered_paragraph_first:
        tokenized = default_tokenizer()(sent, truncation=True, max_length=MAX_SENTENCE_TOKENS, add_special_tokens=False)['input_ids']
        inputs_ids.extend(tokenized)
    inputs_ids.append(default_tokenizer().sep_token_id)
    token_type_ids.extend([0] * (len(inputs_ids)))
    for sent in ordered_paragraph_second:
        tokenized = default_tokenizer()(sent, truncation=True, max_length=MAX_SENTENCE_TOKENS, add_special_tokens=False)['input_ids']
        inputs_ids.extend(tokenized)
    inputs_ids.append(default_tokenizer().sep_token_id)
    token_type_ids.extend([1] * (len(inputs_ids) - len(token_type_ids)))
    attention_mask = [1] * len(inputs_ids)
    return inputs_ids, attention_mask, token_type_ids

def dataset_filtered(split = 'train'):
    dataset = json.load(open(f'./temp_datasets/{split}_two_pass_results.json', 'r'))
    # val_set = json.load(open('./temp_datasets/val_two_pass_results.json', 'r'))
    items = []
    skip_count_tau_equal = 0
    skip_count_match_true_label = 0
    for item in dataset:
        predicted_label_first = item['predicted_label_first']
        predicted_label_second = item['predicted_label_second']
        true_label = item['true_label']
        if list_equal(predicted_label_first, predicted_label_second):
            continue # 如果一致，不需要训练critic bert
        if list_equal(predicted_label_first, true_label):
            skip_count_match_true_label += 1
            continue # 如果某一个预测完全正确了，也不需要训练critic bert
        elif list_equal(predicted_label_second, true_label):
            skip_count_match_true_label += 1
            continue # 如果某一个预测完全正确了，也不需要训练critic bert
        # 计算两者的性能差异
        tau1 = cal_tau(predicted_label_first, true_label)
        tau2 = cal_tau(predicted_label_second, true_label)
        # 【重要修复】：如果两个预测的 Tau 值一样，Critic 无法判断谁好，直接跳过该样本
        if abs(tau1 - tau2) < 1e-3: # 如果两者的tau值差异小于1e-6，认为是一样的
            # print(f"Skipping item due to identical tau values")
            skip_count_tau_equal += 1
            continue
        if tau1 > tau2:
            item['higher_preds'] = predicted_label_first
        else:
            item['higher_preds'] = predicted_label_second
        items.append(item)
    print(f"Skipped {skip_count_tau_equal} items due to identical tau values.")
    print(f"Skipped {skip_count_match_true_label} items due to matching true label.")
    return items

def train():
    model = CriticBert()
    model.to(DEVICE)
    model.train()
    
    # 因为修复了梯度放大问题，合理的 BERT 微调学习率建议在 1e-5 ~ 3e-5
    optimizer_groups = [
        {"params": model.bert.parameters(), "lr": 2e-5},  
        {"params": model.head.parameters(), "lr": 1e-3} 
    ]
    optimizer = torch.optim.AdamW(optimizer_groups)
    
    # 使用二分类交叉熵损失（自带数值稳定优化，防饱和）
    criterion = nn.BCEWithLogitsLoss()
    
    train_set = dataset_filtered('train')
    writer = common.get_writer()
    
    batch_size_target = 16
    accumulated_counter = 0
    batch_loss = []
    
    optimizer.zero_grad() # 在循环外先清空一次
    for epoch in range(1):
        for item in tqdm(train_set, desc="Training CriticBert"):
            paragraph = item['paragraph']
            true_label = item['true_label']
            predicted_labels = item['higher_preds']
            label = 1 # true_label放前面
            # 数据增强：随机交换位置
            if random.random() < 0.5:
                true_label, predicted_labels = predicted_labels, true_label
                label = 0 # predicted_labels放前面
            label_tensor = torch.tensor([[label]], dtype=torch.float).to(DEVICE)
            
            # 准备输入
            inputs_ids, attention_mask, token_type_ids = bert_input_critic_bert(paragraph, true_label, predicted_labels)
            inputs_ids = torch.tensor(inputs_ids).unsqueeze(0).to(DEVICE) 
            attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(DEVICE) 
            token_type_ids = torch.tensor(token_type_ids).unsqueeze(0).to(DEVICE) 
            
            # 前向传播得到 logits
            logits = model(inputs_ids, attention_mask, token_type_ids) 
            
            # 【重要修复】：计算损失时除以 16，实现真正、平滑的梯度累加（Mean Gradient Scaling）
            loss = criterion(logits, label_tensor) / batch_size_target
            loss.backward()
            
            batch_loss.append(loss.item() * batch_size_target) # 还原真实 loss 用于统计
            accumulated_counter += 1
            
            # 凑满一个完整的 Batch，更新一次参数
            if accumulated_counter == batch_size_target:
                optimizer.step()
                optimizer.zero_grad()
                accumulated_counter = 0
                
                writer.add_scalar(f'Loss', np.mean(batch_loss), writer.global_step)
                batch_loss = []
                writer.global_step += 1
                
        # 处理末尾不满 16 个样本残余的梯度
        if accumulated_counter > 0:
            optimizer.step()
            optimizer.zero_grad()
        save_checkpoint(model, prefix='critic_bert_with_truth', suffix='_epoch_{}'.format(epoch))
    return valid_trained(model)

def valid_trained(model = None):
    if model is None:
        model = CriticBert()
        load_checkpoint(model, 'checkpoints/critic_bert_with_truth_e0.pth')
    model.to(DEVICE)
    model.eval()
    val_set = dataset_filtered('val')
    labels = []
    predicts = []
    for item in tqdm(val_set, desc="Validating CriticBert"):
        paragraph = item['paragraph']
        true_label = item['true_label']
        predicted_label_first = item['predicted_label_first']
        predicted_label_second = item['predicted_label_second']
        if list_equal(predicted_label_first, predicted_label_second):
            continue # 如果一致，不需要验证critic bert
        else:
            tau1 = cal_tau(predicted_label_first, true_label)
            tau2 = cal_tau(predicted_label_second, true_label)
            label = 1 if tau1 > tau2 else 0
            inputs_ids, attention_mask, token_type_ids = bert_input_critic_bert(paragraph, predicted_label_first, predicted_label_second)
            inputs_ids = torch.tensor(inputs_ids).unsqueeze(0).to(DEVICE) # [1, seq_len]
            attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(DEVICE) # [1, seq_len]
            token_type_ids = torch.tensor(token_type_ids).unsqueeze(0).to(DEVICE) # [1, seq_len]
            with torch.no_grad():
                logits = model(inputs_ids, attention_mask, token_type_ids)
            predict = 1 if logits.item() > 0.0 else 0
            predicts.append(predict)
            labels.append(label)
    from sklearn.metrics import f1_score, recall_score, precision_score
    f1 = f1_score(labels, predicts)
    recall = recall_score(labels, predicts)
    precision = precision_score(labels, predicts)
    print(f"F1: {f1}, Recall: {recall}, Precision: {precision}")
    return predicts, labels