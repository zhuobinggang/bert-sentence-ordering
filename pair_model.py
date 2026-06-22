# 听gemini的，直接用排序分数矩阵来推算最佳排序
# NOTE: 在句子对模型中，valid的时候使用句子对顺序头的输出来决定句子位置
from aux_loss import *
from sind import *
from itertools import combinations, permutations
import torch
import itertools
import math
import numpy as np

def get_all_index_pairs(lst):
    pairs = []
    for idx1, idx2 in permutations(range(len(lst)), 2):
        pairs.append((idx1, idx2))
    return pairs

def transform_to_label_format(best_order):
    """
    将 [2, 4, 3, 5, 1] (位置->句子ID) 
    转换为 [5, 1, 3, 2, 4] (句子ID->位置)
    """
    # 初始化一个长度为 5 的标准标签列表
    converted_pred = [0] * len(best_order)
    
    # best_order 里面是 1-indexed 的句子ID
    for position_idx, sentence_id in enumerate(best_order):
        # position_idx 是 0~4，对应的位置数字是 position_idx + 1
        # sentence_id 是 1~5，对应的索引是 sentence_id - 1
        converted_pred[sentence_id - 1] = position_idx + 1
        
    return converted_pred

# 先检查所有MASK对的预测结果
class PairLossBertV2(AuxLossBert):
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
                pair_emb = self.pair_embedding(mask_embs[idx1], mask_embs[idx2]) # size: [hidden_size * 2]
                score = self.pair_classifier(pair_emb) # size: [1]
                if abs(score.item() - pair_label) < 0.5:
                    # print_only_once(labels_batch, idx1, idx2, label1, label2, pair_label, score.item(), input_ids = input_ids[batch])
                    accs.append(1)
                else:
                    accs.append(0)
        avg_accuracy = sum(accs) / len(accs)
        return avg_accuracy
    
    def predict_pair_order_matrix_in_paragraph(self, input_ids, attention_mask, labels):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)  # 获取BERT的输出，直接使用最后一层的CLS向量进行分
        last_hidden_state = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        # 取出[MASK]位置的向量
        mask_token_bool = (input_ids == default_tokenizer().mask_token_id) # [batch_size, seq_len]
        taus = []
        accs = []
        pmrs = []
        for batch in range(last_hidden_state.size(0)): # 对于每个batch，取出所有mask对应的位置
            score_matrix = np.zeros((5, 5)) # 初始化一个5x5的矩阵来存储句子对的前后关系得分
            # print(default_tokenizer().decode(input_ids[batch]))
            mask_bool_batch = mask_token_bool[batch] # [seq_len]
            mask_indices = torch.where(mask_bool_batch)[0] # [num_masks]
            labels_batch = labels[batch][mask_bool_batch] # [num_masks]
            labels_batch = [reverse_indexs_tokenized()[label.item()] for label in labels_batch] # 将token_id转换回标签索引
            # print(labels_batch)
            mask_embs = last_hidden_state[batch][mask_bool_batch] # [num_masks, hidden_size]
            for idx1, idx2 in get_all_index_pairs(mask_indices):
                # print(pair_label)
                pair_emb = self.pair_embedding(mask_embs[idx1], mask_embs[idx2]) # size: [hidden_size * 2]
                score = self.pair_classifier(pair_emb) # size: [1]
                score_matrix[idx1][idx2] = score.item()
            # print("score_matrix:\n", score_matrix)
            best_order = get_best_order_by_enumeration(score_matrix)
            # best_order = get_best_order_by_enumeration_v2(score_matrix)
            best_order = add_one(best_order) # 将0-4的索引转换成1-5的标签
            best_order = transform_to_label_format(best_order) # 将位置->句子ID的格式转换成句子ID->位置的格式
            # print("best_order:", best_order)
            # print("labels:", labels_batch)
            taus.append(cal_tau(best_order, labels_batch))
            accs.append(cal_acc(best_order, labels_batch))
            pmrs.append(cal_PMR(best_order, labels_batch))
        avg_tau = sum(taus) / len(taus)
        avg_acc = sum(accs) / len(accs)
        avg_pmr = sum(pmrs) / len(pmrs)
        return avg_tau, avg_acc, avg_pmr

    
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


def get_best_order_by_enumeration(prob_matrix):
    """
    通过穷举 5! = 120 种全排列，寻找全局两两关系总分最高的最佳句子顺序。
    
    参数:
    prob_matrix (list of list): 5x5 的二维列表，matrix[i][j] 表示第 i 个句子在第 j 个句子前面的概率。
    
    返回:
    list: 使得全局概率对数和最大的句子索引排列。
    """
    n = len(prob_matrix)
    best_score = -float('inf')
    best_perm = None
    
    # eps 用于防止 log(0) 报错
    eps = 1e-9 
    
    # 1. 穷举所有可能的排列组合（对于 5 个句子，一共 120 种）
    for perm in itertools.permutations(range(n)):
        current_log_likelihood = 0.0
        
        # 2. 计算当前排列顺序下的全局总得分
        # perm 的结构类似于 (2, 0, 1, 4, 3)，代表句子的先后顺序
        for a in range(n):
            for b in range(a + 1, n):
                # 在当前假设的顺序中，句子 i 排在句子 j 的前面
                i = perm[a]
                j = perm[b]
                
                # 累加对数概率：模型认为 i 确实应该在 j 前面的概率
                # 使用 log 可以让概率相乘变成得分相加，数学上更严谨
                current_log_likelihood += math.log(prob_matrix[i][j] + eps)
                
        # 3. 寻找使全局联合概率最大的那个排列
        if current_log_likelihood > best_score:
            best_score = current_log_likelihood
            best_perm = perm
            
    return list(best_perm)



def get_best_order_by_enumeration_v2(prob_matrix):
    """
    通过穷举 5! = 120 种全排列，寻找全局两两关系总分最高的最佳句子顺序。
    
    参数:
    prob_matrix (list of list): 5x5 的二维列表，matrix[i][j] 表示第 i 个句子在第 j 个句子前面的概率。
    
    返回:
    list: 使得全局概率对数和最大的句子索引排列。
    """
    n = len(prob_matrix)
    best_score = 0
    best_perm = None

        # 1. 穷举所有可能的排列组合（对于 5 个句子，一共 120 种）
    for perm in itertools.permutations(range(n)):
        current_likelihood = 0.0
        
        # 2. 计算当前排列顺序下的全局总得分
        # perm 的结构类似于 (2, 0, 1, 4, 3)，代表句子的先后顺序
        for a in range(n-1):
            b = a + 1
            # 在当前假设的顺序中，句子 i 排在句子 j 的前面
            i = perm[a]
            j = perm[b]
            # 累加对数概率：模型认为 i 确实应该在 j 前面的概率
            # 使用 log 可以让概率相乘变成得分相加，数学上更严谨
            current_likelihood += prob_matrix[i][j]
                
        # 3. 寻找使全局联合概率最大的那个排列
        if current_likelihood > best_score:
            best_score = current_likelihood
            best_perm = perm
            
    return list(best_perm)

# 使用句子对排序头来预测句子对前后关系，然后组合成最终的排序结果，计算准确率和tau值
def valid_by_pair_head_batched(model = None, split = 'val', split_length = None, dataloader = None):
    model.eval()
    toker = default_tokenizer()
    # 首先将val数据集转换成BertInput格式
    # paragraphs = sind_only_texts_get_by_split('val')[:100] # 取前100个故事进行测试
    if dataloader is None:
        paragraphs = sind_paragraphs(split)
        if split_length is not None:
            common.print_once(f"只使用{split}前{split_length}个故事进行验证")
            paragraphs = paragraphs[:split_length]
        bert_inputs = sind_data_prepare(paragraphs)
        dataloader = bert_inputs_to_dataloader_shuffle(bert_inputs)
    # 然后使用默认的BERT模型进行解码，计算准确率和tau值
    taus = []
    accs = []
    pmrs = []
    for batch in tqdm(dataloader, desc="Validation"):
        input_ids, attention_mask, label_ids = batch
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        label_ids = label_ids.to(DEVICE)
        tau, acc, pmr = model.predict_pair_order_matrix_in_paragraph(input_ids, attention_mask, label_ids)
        taus.append(tau)
        accs.append(acc)
        pmrs.append(pmr)
    avg_tau = sum(taus) / len(taus)
    avg_acc = sum(accs) / len(accs)
    avg_pmr = sum(pmrs) / len(pmrs)
    print(f"Average Tau: {avg_tau:.4f}, Average Accuracy: {avg_acc:.4f}, Average PMR: {avg_pmr:.4f}")
    return avg_tau, avg_acc, avg_pmr

@torch.no_grad()
def test_trained_by_pair_head(the_path = ''):
    model = PairLossBertV2()
    the_path = the_path or './checkpoints/SIND_best_20260616_132444_815731pair_loss_bert_best_acc.pth'
    load_checkpoint(model, the_path)
    model.to(DEVICE)
    model.eval()
    val_dataloader = default_test_dataloader_provider()
    return valid_by_pair_head_batched(model, dataloader=val_dataloader)

@torch.no_grad()
def test_trained_by_mlm_head(the_path = ''):
    model = PairLossBertV2()
    the_path = the_path or './checkpoints/SIND_best_20260616_132444_815731pair_loss_bert_best_acc.pth'
    load_checkpoint(model, the_path)
    model.to(DEVICE)
    model.eval()
    val_dataloader = default_test_dataloader_provider()
    return valid_bert_batched(model.bert, dataloader=val_dataloader)