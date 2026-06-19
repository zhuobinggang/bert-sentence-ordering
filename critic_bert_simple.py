import random
import common
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
from sind import save_checkpoint, load_checkpoint
# 假设你的基础函数库
from two_pass_plus import default_bert, default_tokenizer, DEVICE, sind_only_texts_get_by_split

MAX_SENTENCE_TOKENS = 50

class CriticBert(nn.Module):
    def __init__(self):
        super(CriticBert, self).__init__()
        self.bert = default_bert()
        self.head = nn.Linear(self.bert.config.hidden_size, 1) # 纯线性输出，不加 Sigmoid

    def forward(self, input_ids, attention_mask):
        # 最标准的单文本分类输入
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True) 
        cls_hidden = outputs.hidden_states[-1][:, 0, :] # 提取最后一层 CLS 向量
        logits = self.head(cls_hidden) # [batch_size, 1]
        return logits

def build_bert_input(paragraph_list):
    """ 将5个句子按当前顺序拼接成一条标准的 BERT 输入 """
    input_ids = [default_tokenizer().cls_token_id]
    for sent in paragraph_list:
        tokenized = default_tokenizer()(sent, truncation=True, max_length=MAX_SENTENCE_TOKENS, add_special_tokens=False)['input_ids']
        input_ids.extend(tokenized)
    input_ids.append(default_tokenizer().sep_token_id)
    
    attention_mask = [1] * len(input_ids)
    return input_ids, attention_mask

def create_corrupted_paragraph(clean_paragraph):
    """ 随机挑选两个不相同的位置进行调换，生成一个错误的负样本 """
    corrupted = clean_paragraph.copy()
    idx1, idx2 = random.sample(range(len(clean_paragraph)), 2)
    corrupted[idx1], corrupted[idx2] = corrupted[idx2], corrupted[idx1]
    return corrupted

def train():
    model = CriticBert()
    model.to(DEVICE)
    model.train()
    writer = common.get_writer()
    # 标准微调设置
    optimizer = torch.optim.AdamW([
        {"params": model.bert.parameters(), "lr": 2e-5},  
        {"params": model.head.parameters(), "lr": 1e-3} 
    ])
    criterion = nn.BCEWithLogitsLoss()
    
    # 直接读取 SIND 原文中的正确段落数据 (假设列表内每个元素都是已按正确顺序排好序的 5句话列表)
    # 如果读取出来的是无序的，请务必先根据 true_label 还原成原文章正确的顺序！
    correct_paragraphs = sind_only_texts_get_by_split('train') 
    
    batch_size = 8 # 每次处理8个原始段落（16个输入样本）
    accumulated_loss = []
    optimizer.zero_grad()
    
    for i, tgt_paragraph in enumerate(tqdm(correct_paragraphs, desc="Training Critic Pointwise")):
        # 1. 构造正样本 (Label = 1.0)
        pos_ids, pos_mask = build_bert_input(tgt_paragraph)
        
        # 2. 构造负样本 (随机 Swap 两个句子, Label = 0.0)
        neg_paragraph = create_corrupted_paragraph(tgt_paragraph)
        neg_ids, neg_mask = build_bert_input(neg_paragraph)
        
        # 3. 将它们打包输入（这里为了绝对安全，我们用串行前向，并行反向）
        # 正样本前向
        pos_ids_t = torch.tensor(pos_ids).unsqueeze(0).to(DEVICE)
        pos_mask_t = torch.tensor(pos_mask).unsqueeze(0).to(DEVICE)
        pos_label = torch.tensor([[1.0]], dtype=torch.float).to(DEVICE)
        pos_logits = model(pos_ids_t, pos_mask_t)
        loss_pos = criterion(pos_logits, pos_label) / (batch_size * 2)
        loss_pos.backward()
        
        # 负样本前向
        neg_ids_t = torch.tensor(neg_ids).unsqueeze(0).to(DEVICE)
        neg_mask_t = torch.tensor(neg_mask).unsqueeze(0).to(DEVICE)
        neg_label = torch.tensor([[0.0]], dtype=torch.float).to(DEVICE)
        neg_logits = model(neg_ids_t, neg_mask_t)
        loss_neg = criterion(neg_logits, neg_label) / (batch_size * 2)
        loss_neg.backward()
        
        accumulated_loss.append(loss_pos.item() * batch_size * 2 + loss_neg.item() * batch_size * 2)
        
        # 满足 Batch Size 更新梯度
        if (i + 1) % batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()
            writer.add_scalar('Loss', np.mean(accumulated_loss), writer.global_step)
            accumulated_loss = []
            writer.global_step += 1

    # 挽底更新
    optimizer.step()
    optimizer.zero_grad()
    
    save_checkpoint(model, prefix='critic_bert', suffix='pointwise_good')
    return valid_trained(model)


def valid_trained(model = None):
    if model is None:
        model = CriticBert()
        load_checkpoint(model, 'checkpoints/critic_bert_pointwise_good.pth')
    model.to(DEVICE)
    model.eval()
    
    val_paragraphs = sind_only_texts_get_by_split('val')
    correct_count = 0
    total_count = 0
    
    for tgt_paragraph in tqdm(val_paragraphs, desc="Validating Pointwise Critic"):
        # 生成正负输入
        pos_ids, pos_mask = build_bert_input(tgt_paragraph)
        neg_paragraph = create_corrupted_paragraph(tgt_paragraph)
        neg_ids, neg_mask = build_bert_input(neg_paragraph)
        
        pos_ids_t = torch.tensor(pos_ids).unsqueeze(0).to(DEVICE)
        pos_mask_t = torch.tensor(pos_mask).unsqueeze(0).to(DEVICE)
        
        neg_ids_t = torch.tensor(neg_ids).unsqueeze(0).to(DEVICE)
        neg_mask_t = torch.tensor(neg_mask).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            pos_score = model(pos_ids_t, pos_mask_t).item()
            neg_score = model(neg_ids_t, neg_mask_t).item()
            
        # 只要正样本的连贯性得分高于负样本，说明模型判断正确
        if pos_score > neg_score:
            correct_count += 1
        total_count += 1
        
    acc = correct_count / total_count if total_count > 0 else 0
    print(f"Critic 判别准确率 (Pairwise Accuracy on Val): {acc:.4f}")
    return acc