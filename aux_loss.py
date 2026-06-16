# 使用[MASK]向量来判断句子对的前后关系
from bert_utils import DEVICE, default_bert, default_tokenizer, reverse_indexs_tokenized
from sind import sind_data_prepare, train
from torch import nn
import torch
from recordclass import recordclass

PairLossBertResult = recordclass('PairLossBertResult', 'loss decode_loss pair_loss')

def print_only_once(*args, input_ids=None):
    if not hasattr(print_only_once, "has_printed"):
        print(default_tokenizer().decode(input_ids))
        for arg in args:
            print(arg)
        print_only_once.has_printed = True

class PairLossBert(nn.Module):
    def __init__(self, bert=None):
        super(PairLossBert, self).__init__()
        if bert is None:
            bert = default_bert()
        self.bert = bert
        # linear一个线性层接一个sigmoid层，1代表前后关系正确，0代表前后关系错误
        self.pair_classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, labels=labels)  # 获取BERT的输出，直接使用最后一层的CLS向量进行分
        decode_loss = outputs.loss
        last_hidden_state = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        # 取出[MASK]位置的向量
        mask_token_bool = (input_ids == default_tokenizer().mask_token_id) # [batch_size, seq_len]
        classification_loss = 0.0
        for batch in range(last_hidden_state.size(0)): # 对于每个batch，取出所有mask对应的位置
            # print(default_tokenizer().decode(input_ids[batch]))
            mask_bool_batch = mask_token_bool[batch] # [seq_len]
            mask_indices = torch.where(mask_bool_batch)[0] # [num_masks]
            labels_batch = labels[batch][mask_bool_batch] # [num_masks]
            # print(labels_batch)
            mask_embs = last_hidden_state[batch][mask_bool_batch] # [num_masks, hidden_size]
            # 随机取出两个mask位置的向量进行拼接，送入线性层进行二分类
            idx1, idx2 = torch.randperm(len(mask_indices))[:2] # �
            # print(idx1, idx2)
            # 根据labels判断idx1和idx2的前后关系，构造二分类标签
            label1 = reverse_indexs_tokenized()[labels_batch[idx1].item()] # 将token_id转换回标签索引
            label2 = reverse_indexs_tokenized()[labels_batch[idx2].item()]
            # print(label1, label2)
            pair_label = 1 if label1 < label2 else 0 # 如果label1在label2前面，标签为1，否则为0
            # print(pair_label)
            pair_emb = torch.cat([mask_embs[idx1], mask_embs[idx2]], dim=-1) # size: [hidden_size * 2]
            score = self.pair_classifier(pair_emb) # size: [1]
            # print(score.item())
            print_only_once(labels_batch, idx1, idx2, label1, label2, pair_label, score.item(), input_ids = input_ids[batch])
            square_loss = (score - pair_label) ** 2
            classification_loss += square_loss
        classification_loss = classification_loss / last_hidden_state.size(0) # 平均每个batch的损失
        loss = decode_loss + classification_loss
        return PairLossBertResult(loss=loss, decode_loss=decode_loss.item(), pair_loss=classification_loss.item())
    
    def predict_pair_order(self, s1, s2):
        # 输入两个句子，判断它们的前后关系，返回s1在s2前面的概率
        bert_input = sind_data_prepare([[s1, s2]], need_shuffle=False)[0]
        input_ids, attention_mask = torch.tensor([bert_input.input_ids]).to(DEVICE), torch.tensor([bert_input.attention_mask]).to(DEVICE)
        outputs = self.bert(input_ids=input_ids, 
                            attention_mask=attention_mask, 
                            output_hidden_states=True)  # 获取BERT的输出，直接使用最后一层的CLS向量进行分类
        last_hidden_state = outputs.hidden_states[-1]  # [1, seq_len, hidden_size]
        mask_token_bool = (input_ids == default_tokenizer().mask_token_id) # [seq_len]
        mask_indices = torch.where(mask_token_bool)[0] # [num_masks]
        mask_embs = last_hidden_state[0][mask_indices] # [num_masks, hidden_size]
        pair_emb = torch.cat([mask_embs[0], mask_embs[1]], dim=-1) # size: [hidden_size * 2]
        score = self.pair_classifier(pair_emb) # size: [1]
        return score.item() # 返回前后关系的概率，越接近1表示s1在s2前面，越接近0表示s2在s1前面
    
    def pair_order_loss(self, s1, s2, label):
        bert_input = sind_data_prepare([[s1, s2]], need_shuffle=False)[0]
        input_ids, attention_mask = torch.tensor([bert_input.input_ids]).to(DEVICE), torch.tensor([bert_input.attention_mask]).to(DEVICE)
        outputs = self.bert(input_ids=input_ids, 
                            attention_mask=attention_mask, 
                            output_hidden_states=True)  # 获取BERT的输出，直接使用最后一层的CLS向量进行分类
        last_hidden_state = outputs.hidden_states[-1]  # [1, seq_len, hidden_size]
        mask_token_bool = (input_ids == default_tokenizer().mask_token_id) # [seq_len]
        mask_indices = torch.where(mask_token_bool)[0] # [num_masks]
        mask_embs = last_hidden_state[0][mask_indices] # [num_masks, hidden_size]
        pair_emb = torch.cat([mask_embs[0], mask_embs[1]], dim=-1) # size: [hidden_size * 2]
        score = self.pair_classifier(pair_emb) # size: [1]
        square_loss = (score - label) ** 2
        return square_loss
    

def train_pair_loss_bert():
    model = PairLossBert()
    model.to(DEVICE)
    train(epochs=5, model=model, suffix='pair_loss_bert')