import random
import common
import rocs
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
from sind import save_checkpoint, load_checkpoint
# 假设你的基础函数库
from two_pass_plus import default_bert, default_tokenizer, DEVICE, sind_paragraphs

MAX_SENTENCE_TOKENS = 50

class CriticBert(nn.Module):
    def __init__(self):
        super(CriticBert, self).__init__()
        self.bert = default_bert()
        # 输出每个句子对应的排序得分（分数越高，代表在文章中越靠前）
        self.head = nn.Linear(self.bert.config.hidden_size, 1) 

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True) 
        last_hidden_state = outputs.hidden_states[-1] # [batch_size, seq_len, hidden_size]
        
        # 动态提取所有 [CLS] token 的位置
        cls_token_id = default_tokenizer().cls_token_id
        cls_indices = (input_ids == cls_token_id).nonzero(as_tuple=True)
        
        # 提取出每个句子开头的 CLS 向量
        cls_hidden = last_hidden_state[cls_indices[0], cls_indices[1], :] # [num_sentences, hidden_size]
        
        logits = self.head(cls_hidden) # [num_sentences, 1]
        return logits.squeeze(-1) # [num_sentences]

def build_bert_input(paragraph_list):
    """ 将所有句子拼接，并且在【每个句子】前面都添加一个 [CLS] token """
    input_ids = []
    cls_token_id = default_tokenizer().cls_token_id
    for sent in paragraph_list:
        input_ids.append(cls_token_id) 
        tokenized = default_tokenizer()(sent, truncation=True, max_length=MAX_SENTENCE_TOKENS, add_special_tokens=False)['input_ids']
        input_ids.extend(tokenized)
    input_ids.append(default_tokenizer().sep_token_id) 
    
    attention_mask = [1] * len(input_ids)
    return input_ids, attention_mask

def create_shuffled_paragraph_with_labels(clean_paragraph):
    """ 
    将完美顺序的段落随机彻底打乱，并返回打乱后的段落以及每个句子对应的【真实绝对位置标签】。
    例如：原段落为 [A, B, C] (正确顺序)
    打乱后可能为 [C, A, B]
    对应的 true_positions 就是 [2.0, 0.0, 1.0] (因为C原本是第2句，A是第0句，B是第1句)
    """
    indexed_paragraph = list(enumerate(clean_paragraph))
    random.shuffle(indexed_paragraph) # 彻底打乱
    
    shuffled_paragraph = [item[1] for item in indexed_paragraph]
    true_positions = [float(item[0]) for item in indexed_paragraph]
    return shuffled_paragraph, true_positions

def listmle_loss(scores, true_positions):
    """
    ListMLE 损失函数实现
    scores: [num_sentences] - 模型预测的得分
    true_positions: [num_sentences] - 真实的绝对位置（值越小越靠前）
    目标：使真正靠前（true_position小）的句子，拥有更高的预测分数（score大）
    """
    # 1. 根据真实位置进行升序排列的索引 (让原本排在第0位的句子排在最前面)
    indices = torch.argsort(true_positions)
    ordered_scores = scores[indices] # 排序后的分数向量 [num_sentences]
    
    # 2. 计算 ListMLE 的负对数似然
    loss = 0.0
    n = ordered_scores.size(0)
    for i in range(n):
        # 当前位置及之后的所有分数
        sub_scores = ordered_scores[i:]
        # ListMLE 公式：log(sum(exp(sub_scores))) - ordered_scores[i]
        loss += torch.logsumexp(sub_scores, dim=0) - ordered_scores[i]
    return loss

def get_critic_score(critic_model, paragraph):
    """ 
    输入一个段落，返回其整体连贯性得分。
    我们假设当前输入的顺序就是“正确的”，计算其在当前状态下的 ListMLE 损失。
    损失越小，说明当前顺序越符合逻辑。取【负损失】作为得分，得分越高（越接近0）越连贯。
    """
    input_ids, attention_mask = build_bert_input(paragraph)
    input_ids_t = torch.tensor(input_ids).unsqueeze(0).to(DEVICE)
    attention_mask_t = torch.tensor(attention_mask).unsqueeze(0).to(DEVICE)
    
    # 评估当前序列，因此假设理想的位置就是 [0, 1, 2, 3...]
    ideal_positions = torch.arange(len(paragraph), dtype=torch.float).to(DEVICE)
    
    with torch.no_grad():
        scores = critic_model(input_ids_t, attention_mask_t)
        loss = listmle_loss(scores, ideal_positions)
        score = -loss.item() # 负损失作为 Coherence Score
    return score

def train(sind = True, epoch = 5):
    model = CriticBert()
    model.to(DEVICE)
    model.train()
    writer = common.get_writer()
    
    optimizer = torch.optim.AdamW([
        {"params": model.bert.parameters(), "lr": 2e-5},  
        {"params": model.head.parameters(), "lr": 1e-3} 
    ])
    
    correct_paragraphs = sind_paragraphs('train') if sind else rocs.dataset_get()['train']
    prefix = 'critic_bert' + ('_sind' if sind else '_rocs')
    
    batch_size = 8 # 累积 8 个段落更新一次梯度
    accumulated_loss = []
    optimizer.zero_grad()
    best_acc = 0.0
    
    for epoch_idx in range(epoch):
        for i, tgt_paragraph in enumerate(tqdm(correct_paragraphs, desc=f"Epoch {epoch_idx} | Training Critic ListMLE")):
            # 1. 彻底打乱输入，并拿到它在原文章中的真实绝对位置
            shuffled_paragraph, true_positions_list = create_shuffled_paragraph_with_labels(tgt_paragraph)
            
            # 2. 构造 BERT 输入并前向传播
            input_ids, attention_mask = build_bert_input(shuffled_paragraph)
            input_ids_t = torch.tensor(input_ids).unsqueeze(0).to(DEVICE)
            attention_mask_t = torch.tensor(attention_mask).unsqueeze(0).to(DEVICE)
            true_positions_t = torch.tensor(true_positions_list, dtype=torch.float).to(DEVICE)
            
            logits = model(input_ids_t, attention_mask_t)
            
            # 3. 计算 ListMLE 损失
            loss = listmle_loss(logits, true_positions_t) / batch_size
            loss.backward()
            
            accumulated_loss.append(loss.item() * batch_size)
            
            # 梯度累积更新
            if (i + 1) % batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()
                writer.add_scalar('Loss', np.mean(accumulated_loss), writer.global_step)
                accumulated_loss = []
                writer.global_step += 1
                
        # 挽底更新
        optimizer.step()
        optimizer.zero_grad()
        
        # 验证当前 Epoch 的判别准确率
        acc = valid_trained(model)
        if acc > best_acc:
            print(f"New best accuracy: {acc:.4f}, save model checkpoint")
            best_acc = acc
            save_checkpoint(model, prefix=prefix, suffix='listmle_best')

def default_critic_model_sind():
    model = CriticBert()
    load_checkpoint(model, 'checkpoints/critic_sind_default.pth')
    model.to(DEVICE)
    model.eval()
    return model

def default_critic_model_rocs():
    model = CriticBert()
    load_checkpoint(model, 'checkpoints/critic_rocs_default.pth')
    model.to(DEVICE)
    model.eval()
    return model

def valid_trained(model = None):
    if model is None:
        model = default_critic_model_sind()
    model.to(DEVICE)
    model.eval()
    
    val_paragraphs = sind_paragraphs('val')
    correct_count = 0
    total_count = 0
    
    for tgt_paragraph in tqdm(val_paragraphs, desc="Validating ListMLE Critic"):
        # 正样本得分（标准顺序的负 ListMLE Loss）
        pos_score = get_critic_score(model, tgt_paragraph)
        
        # 负样本得分（彻底打乱顺序后的负 ListMLE Loss）
        shuffled_paragraph, _ = create_shuffled_paragraph_with_labels(tgt_paragraph)
        neg_score = get_critic_score(model, shuffled_paragraph)
        
        # 只要标准顺序的得分高于打乱顺序的得分，说明模型具备正确的排序分辨能力
        if pos_score > neg_score:
            correct_count += 1
        total_count += 1
        
    acc = correct_count / total_count if total_count > 0 else 0
    print(f"Critic 判别准确率 (Pairwise Accuracy on Val): {acc:.4f}")
    return acc