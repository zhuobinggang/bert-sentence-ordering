# 使用[MASK]向量来判断句子对的前后关系
from bert_utils import DEVICE, default_bert, default_tokenizer, reverse_indexs_tokenized
from sind import sind_data_prepare, train, load_checkpoint, default_val_dataloader_provider, valid_bert_batched, default_test_dataloader_provider
from torch import nn
import torch
from recordclass import recordclass
from tqdm import tqdm

PairLossBertResult = recordclass('PairLossBertResult', 'loss decode_loss pair_loss')
DOUBLE_CHECK = True # 是否开启训练时的双重检查，反过来训练一次，增加训练信号

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
        self.init_pair_classifier()

    def init_pair_classifier(self):
        # linear一个线性层接一个sigmoid层，1代表前后关系正确，0代表前后关系错误
        self.pair_classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size * 2, 1),
            nn.Sigmoid()
        )

    def pair_embedding(self, emb1, emb2):
        return torch.cat([emb1, emb2], dim=-1)

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
            pair_emb = self.pair_embedding(mask_embs[idx1], mask_embs[idx2]) # size: [hidden_size * 2]
            score = self.pair_classifier(pair_emb) # size: [1]
            # print(score.item())
            print_only_once(labels_batch, idx1, idx2, label1, label2, pair_label, score.item(), input_ids = input_ids[batch])
            square_loss = (score - pair_label) ** 2
            if DOUBLE_CHECK:
                # 反过来也训练一次
                pair_emb_reverse = self.pair_embedding(mask_embs[idx2], mask_embs[idx1]) # size: [hidden_size * 2]
                score_reverse = self.pair_classifier(pair_emb_reverse) # size: [1]
                square_loss += (score_reverse - (1 - pair_label)) ** 2
            classification_loss += square_loss
        classification_loss = classification_loss / last_hidden_state.size(0) # 平均每个batch的损失
        loss = decode_loss + classification_loss
        return PairLossBertResult(loss=loss, decode_loss=decode_loss.item(), pair_loss=classification_loss.item())
    
    def predict_pair_order_in_paragraph(self, input_ids, attention_mask, labels):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)  # 获取BERT的输出，直接使用最后一层的CLS向量进行分
        last_hidden_state = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        # 取出[MASK]位置的向量
        mask_token_bool = (input_ids == default_tokenizer().mask_token_id) # [batch_size, seq_len]
        avg_accuracy = 0.0
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
            pair_emb = self.pair_embedding(mask_embs[idx1], mask_embs[idx2]) # size: [hidden_size * 2]
            score = self.pair_classifier(pair_emb) # size: [1]
            if abs(score.item() - pair_label) < 0.5:
                # print_only_once(labels_batch, idx1, idx2, label1, label2, pair_label, score.item(), input_ids = input_ids[batch])
                avg_accuracy += 1.0
        avg_accuracy = avg_accuracy / last_hidden_state.size(0) # 平均每个
        return avg_accuracy
    
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
        pair_emb = self.pair_embedding(mask_embs[0], mask_embs[1]) # size: [hidden_size * 2]
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
        pair_emb = self.pair_embedding(mask_embs[0], mask_embs[1]) # size: [hidden_size * 2]
        score = self.pair_classifier(pair_emb) # size: [1]
        square_loss = (score - label) ** 2
        return square_loss
    

def train_pair_loss_bert():
    model = PairLossBert()
    model.to(DEVICE)
    train(epochs=5, model=model, suffix='_pair_loss_bertx2')

def test_trained_for_pair():
    model = PairLossBert()
    load_checkpoint(model, './checkpoints/SIND_best_20260616_132444_815731pair_loss_bert_best_acc.pth')
    model.to(DEVICE)
    model.eval()
    s1 = "The cat is on the mat."
    s2 = "The mat is under the cat."
    score = model.predict_pair_order(s1, s2)
    print(f"Score for '{s1}' before '{s2}': {score:.4f}")
    return model

def test_trained_for_pair():
    model = PairLossBert()
    load_checkpoint(model, './checkpoints/SIND_best_20260616_132444_815731pair_loss_bert_best_acc.pth')
    model.to(DEVICE)
    model.eval()
    val_dataloader = default_val_dataloader_provider()
    avg_acc = 0.0
    for batch in tqdm(val_dataloader, desc="Validation"):
        input_ids, attention_mask, label_ids = batch
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        label_ids = label_ids.to(DEVICE)
        acc = model.predict_pair_order_in_paragraph(input_ids, attention_mask, label_ids)
        # print(f"{acc:.4f}")
        avg_acc += acc
    avg_acc = avg_acc / len(val_dataloader)
    print(f"Average pair order accuracy in validation set: {avg_acc:.4f}")

def test_trained():
    model = PairLossBert()
    load_checkpoint(model, './checkpoints/SIND_best_20260616_132444_815731pair_loss_bert_best_acc.pth')
    model.to(DEVICE)
    model.eval()
    valid_bert_batched(model.bert, split='test')