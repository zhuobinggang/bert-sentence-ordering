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
    paragraphs = sind_only_texts_get_by_split(split)
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
        self.head = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)  # 获取BERT的输出，直接使用最后一层的CLS向量进行分
        last_hidden_state = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        cls_hidden_state = last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        probs = self.head(cls_hidden_state)  # [batch_size, 1]
        return probs

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
    inputs_ids = []
    inputs_ids.append(default_tokenizer().cls_token_id)
    for sent in ordered_paragraph_first:
        tokenized = default_tokenizer()(sent, truncation=True, max_length=MAX_SENTENCE_TOKENS, add_special_tokens=False)['input_ids']
        inputs_ids.extend(tokenized)
    inputs_ids.append(default_tokenizer().sep_token_id)
    for sent in ordered_paragraph_second:
        tokenized = default_tokenizer()(sent, truncation=True, max_length=MAX_SENTENCE_TOKENS, add_special_tokens=False)['input_ids']
        inputs_ids.extend(tokenized)
    inputs_ids.append(default_tokenizer().sep_token_id)
    attention_mask = [1] * len(inputs_ids)
    return inputs_ids, attention_mask

def dataset_filtered(split = 'train'):
    dataset = json.load(open(f'./temp_datasets/{split}_two_pass_results.json', 'r'))
    # val_set = json.load(open('./temp_datasets/val_two_pass_results.json', 'r'))
    items = []
    for item in dataset:
        predicted_label_first = item['predicted_label_first']
        predicted_label_second = item['predicted_label_second']
        if list_equal(predicted_label_first, predicted_label_second):
            continue # 如果一致，不需要训练critic bert
        else:
            items.append(item)
    return items

def train():
    model = CriticBert()
    model.to(DEVICE)
    model.train()
    optimizer_groups = [
        {"params": model.bert.parameters(), "lr": 5e-5},  
        {"params": model.head.parameters(), "lr": 5e-3} 
    ]
    optimizer = torch.optim.AdamW(optimizer_groups)
    train_set = dataset_filtered('train')
    writer = common.get_writer()
    # val_set = json.load(open('./temp_datasets/val_two_pass_results.json', 'r'))
    for item in tqdm(train_set, desc="Training CriticBert"):
        paragraph = item['paragraph']
        true_label = item['true_label']
        predicted_label_first = item['predicted_label_first']
        predicted_label_second = item['predicted_label_second']
        if list_equal(predicted_label_first, predicted_label_second):
            continue # 如果一致，不需要训练critic bert
        else:
            inputs_ids, attention_mask = bert_input_critic_bert(paragraph, predicted_label_first, predicted_label_second)
            inputs_ids = torch.tensor(inputs_ids).unsqueeze(0).to(DEVICE) # [1, seq_len]
            attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(DEVICE) # [1, seq_len]
            # 计算label
            tau1 = cal_tau(predicted_label_first, true_label)
            tau2 = cal_tau(predicted_label_second, true_label)
            label = 1 if tau1 > tau2 else 0
            # 计算loss
            probs = model(inputs_ids, attention_mask) # [1, 1]
            square_loss = (probs - label) ** 2
            writer.add_scalar(f'Loss', square_loss.item(), writer.global_step)
            writer.global_step += 1
            # backward and step
            square_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    save_checkpoint(model, prefix='critic_bert', suffix = 'e0')
    valid_trained(model)

def valid_trained(model = None):
    if model is None:
        model = CriticBert()
        load_checkpoint(model, 'checkpoints/critic_bert_e0.pth')
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
            inputs_ids, attention_mask = bert_input_critic_bert(paragraph, predicted_label_first, predicted_label_second)
            inputs_ids = torch.tensor(inputs_ids).unsqueeze(0).to(DEVICE) # [1, seq_len]
            attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(DEVICE) # [1, seq_len]
            with torch.no_grad():
                probs = model(inputs_ids, attention_mask)
            predict = 1 if probs.item() > 0.5 else 0
            predicts.append(predict)
            labels.append(label)
    from sklearn.metrics import f1_score, recall_score, precision_score
    f1 = f1_score(labels, predicts)
    recall = recall_score(labels, predicts)
    precision = precision_score(labels, predicts)
    print(f"F1: {f1}, Recall: {recall}, Precision: {precision}")
    return f1, recall, precision