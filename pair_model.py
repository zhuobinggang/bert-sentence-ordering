# 听gemini的，直接用排序分数矩阵来推算最佳排序
from aux_loss import *
from itertools import combinations
import torch

def get_all_index_pairs(lst):
    pairs = []
    for idx1, idx2 in combinations(range(len(lst)), 2):
        # pair_emb = torch.cat([mask_embs[idx1], mask_embs[idx2]], dim=-1) # size: [hidden_size * 2]
        pairs.append((idx1, idx2))
    return pairs

# 先检查所有MASK对的预测结果
class PairLossBertV2(PairLossBert):
    def predict_pair_order_in_paragraph(self, input_ids, attention_mask, labels):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)  # 获取BERT的输出，直接使用最后一层的CLS向量进行分
        last_hidden_state = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        # 取出[MASK]位置的向量
        mask_token_bool = (input_ids == default_tokenizer().mask_token_id) # [batch_size, seq_len]
        accs = []
        for batch in range(last_hidden_state.size(0)): # 对于每个batch，取出所有mask对应的位置
            # print(default_tokenizer().decode(input_ids[batch]))
            mask_bool_batch = mask_token_bool[batch] # [seq_len]
            mask_indices = torch.where(mask_bool_batch)[0] # [num_masks]
            labels_batch = labels[batch][mask_bool_batch] # [num_masks]
            # print(labels_batch)
            mask_embs = last_hidden_state[batch][mask_bool_batch] # [num_masks, hidden_size]
            for idx1, idx2 in get_all_index_pairs(mask_indices):
                # 随机取出两个mask位置的向量进行拼接，送入线性层进行二分类
                # idx1, idx2 = torch.randperm(len(mask_indices))[:2] # �
                # print(idx1, idx2)
                # 根据labels判断idx1和idx2的前后关系，构造二分类标签
                label1 = reverse_indexs_tokenized()[labels_batch[idx1].item()] # 将token_id转换回标签索引
                label2 = reverse_indexs_tokenized()[labels_batch[idx2].item()]
                # print(label1, label2)
                pair_label = 1 if label1 < label2 else 0 # 如果label1在label2前面，标签为1，否则为0
                # print(pair_label)
                pair_emb = torch.cat([mask_embs[idx1], mask_embs[idx2]], dim=-1) # size: [hidden_size * 2]
                score = self.pair_classifier(pair_emb) # size: [1]
                if abs(score.item() - pair_label) < 0.5:
                    # print_only_once(labels_batch, idx1, idx2, label1, label2, pair_label, score.item(), input_ids = input_ids[batch])
                    accs.append(1)
                else:
                    accs.append(0)
        avg_accuracy = sum(accs) / len(accs)
        return avg_accuracy
    
@torch.no_grad()
def test_trained_for_pair(the_path = ''):
    model = PairLossBertV2()
    the_path = the_path or './checkpoints/SIND_best_20260616_132444_815731pair_loss_bert_best_acc.pth'
    load_checkpoint(model, the_path)
    model.to(DEVICE)
    model.eval()
    val_dataloader = default_test_dataloader_provider()
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
    print(f"Average pair order accuracy in Test set: {avg_acc:.4f}")