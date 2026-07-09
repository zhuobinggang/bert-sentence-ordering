# 训练NIPS上的critic模型，使用bert作为基础模型
# from critic_bert_simple import *
import common
import critic_bert_simple
import nips_data
import nips_bert_input
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
logger = common.logging.getLogger(__name__)
from critic_bert_simple import CriticBert
from sind import load_checkpoint

def nips_bert_input_for_critic(paragraph):
    bert_input = nips_bert_input.nips_bert_input(paragraph, need_shuffle = False) # NOTE: 这里不能shuffle否则BUG
    return bert_input.input_ids, bert_input.attention_mask


def valid_trained_nips(model = None):
    if model is None:
        model = critic_bert_simple.CriticBert()
        model.to(DEVICE)
        model.eval()

    val_paragraphs = nips_data.get_paragraphs('val')
    correct_count = 0
    total_count = 0
    
    for tgt_paragraph in tqdm(val_paragraphs, desc="Validating Pointwise Critic"):
        # 生成正负输入
        pos_ids, pos_mask = nips_bert_input_for_critic(tgt_paragraph)
        neg_paragraph = critic_bert_simple.create_corrupted_paragraph(tgt_paragraph)
        neg_ids, neg_mask = nips_bert_input_for_critic(neg_paragraph)
        
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

# NOTE: 因为nips数据集很小所以可能需要多个epoch
def train(epoch = 1):
    model = critic_bert_simple.CriticBert()
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
    correct_paragraphs = nips_data.get_paragraphs('train')
    prefix = 'critic_bert' + '_nips'
    
    batch_size = 8 # 每次处理8个原始段落（16个输入样本）
    accumulated_loss = []
    optimizer.zero_grad()
    best_acc = 0.0
    for epoch in range(epoch):
        for i, tgt_paragraph in enumerate(tqdm(correct_paragraphs, desc="Training Critic Pointwise")):
            # 1. 构造正样本 (Label = 1.0)
            pos_ids, pos_mask = nips_bert_input_for_critic(tgt_paragraph)
            
            # 2. 构造负样本 (随机 Swap 两个句子, Label = 0.0)
            neg_paragraph = critic_bert_simple.create_corrupted_paragraph(tgt_paragraph)
            neg_ids, neg_mask = nips_bert_input_for_critic(neg_paragraph)
            
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
        acc = valid_trained_nips(model)
        if acc > best_acc:
            msg = f"New best accuracy: {acc:.4f}, save model checkpoint"
            print(msg)
            logger.warning(msg)
            best_acc = acc
            critic_bert_simple.save_checkpoint(model, prefix=prefix, suffix='')

def default_critic_model_nips():
    model = CriticBert()
    load_checkpoint(model, 'checkpoints/critic_nips_default.pth')
    model.to(DEVICE)
    model.eval()
    return model