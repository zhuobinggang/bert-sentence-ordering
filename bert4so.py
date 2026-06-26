import random
import common
import rocs
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
from sind import save_checkpoint, load_checkpoint, cal_tau, cal_acc, cal_PMR
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

def train(sind=True, num_repeats=3, epochs_per_repeat=5):
    """
    支持多轮独立重复训练（Repeat）的 ListMLE 训练函数。
    每个 Repeat 训练 epochs_per_repeat 次，并根据验证集 Tau 指标保存各自的最佳模型。
    """
    writer = common.get_writer()
    correct_paragraphs = sind_paragraphs('train') if sind else rocs.dataset_get()['train']
    prefix = 'critic_bert' + ('_sind' if sind else '_rocs')
    batch_size = 8 
    
    # ==================== 外层循环：控制训练的 Repeat 次数 ====================
    for repeat_idx in range(num_repeats):
        print(f"\n" + "="*30)
        print(f"🚀 开始第 {repeat_idx + 1} / {num_repeats} 次独立重复训练 (Repeat)")
        print(f"==========" + "="*30)
        
        # 核心：每次 repeat 必须重新实例化模型和优化器，确保从头训练
        model = CriticBert()
        model.to(DEVICE)
        
        optimizer = torch.optim.AdamW([
            {"params": model.bert.parameters(), "lr": 2e-5},  
            {"params": model.head.parameters(), "lr": 1e-3} 
        ])
        
        best_tau = -1.0  # 局部变量，只记录当前这一轮 repeat 中的最佳 Tau
        accumulated_loss = []
        optimizer.zero_grad()
        
        # ==================== 中层循环：控制当前 Repeat 的 Epoch ====================
        for epoch_idx in range(epochs_per_repeat):
            model.train()
            
            # 使用 tqdm 展示当前属于哪个 Repeat 和哪个 Epoch
            desc_str = f"Rep {repeat_idx + 1} | Epoch {epoch_idx + 1}"
            for i, tgt_paragraph in enumerate(tqdm(correct_paragraphs, desc=desc_str)):
                
                # 1. 打乱输入，获取真实绝对位置
                shuffled_paragraph, true_positions_list = create_shuffled_paragraph_with_labels(tgt_paragraph)
                
                # 2. 构造输入并前向传播
                input_ids, attention_mask = build_bert_input(shuffled_paragraph)
                input_ids_t = torch.tensor(input_ids).unsqueeze(0).to(DEVICE)
                attention_mask_t = torch.tensor(attention_mask).unsqueeze(0).to(DEVICE)
                true_positions_t = torch.tensor(true_positions_list, dtype=torch.float).to(DEVICE)
                
                logits = model(input_ids_t, attention_mask_t)
                
                # 3. 计算 ListMLE 损失并反向传播
                loss = listmle_loss(logits, true_positions_t) / batch_size
                loss.backward()
                
                accumulated_loss.append(loss.item() * batch_size)
                
                # 4. 梯度累积与更新
                if (i + 1) % batch_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    # 在 TensorBoard 中区分不同的 repeat 路径
                    writer.add_scalar(f'Loss/Repeat_{repeat_idx + 1}', np.mean(accumulated_loss), writer.global_step)
                    accumulated_loss = []
                    writer.global_step += 1
                    
            # 5. 每个 Epoch 结束时的挽底更新
            optimizer.step()
            optimizer.zero_grad()
            
            # ==================== 内层：每个 Epoch 结束后的验证与保存 ====================
            print(f"\n[Rep {repeat_idx + 1} | Epoch {epoch_idx + 1}] 正在进行验证集全指标评估...")
            # 调用你写好的新评估函数，解包获取 tau
            tau, acc, pmr = valid_trained_sentence_ordering(model, sind=sind)
            
            # 根据当前 repeat 的内部 best_tau 决定是否更新
            if tau > best_tau:
                print(f"🔥 [Repeat {repeat_idx + 1}] 触发最佳权重更新! Tau 从 {best_tau:.4f} 提升至 {tau:.4f}")
                best_tau = tau
                
                # 动态生成包含 repeat 编号的唯一后缀，例如：listmle_rep1_best, listmle_rep2_best...
                current_suffix = f'listmle_rep{repeat_idx + 1}_best'
                save_checkpoint(model, prefix=prefix, suffix=current_suffix)
            else:
                print(f"放回未保存。当前 Tau: {tau:.4f}，当前 Repeat 历史最佳 Tau: {best_tau:.4f}")
    print("\n✅ 所有 Repeat 训练任务已全部完成！")

def default_critic_model_sind():
    model = CriticBert()
    load_checkpoint(model, 'checkpoints/critic_bert_sind_listmle_best.pth')
    model.to(DEVICE)
    model.eval()
    return model

def default_critic_model_rocs():
    model = CriticBert()
    load_checkpoint(model, 'checkpoints/critic_bert_rocs_listmle_best.pth')
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



def valid_trained_sentence_ordering(model=None, sind=True):
    """
    使用训练好的 ListMLE 模型在验证集/测试集上运行，
    并计算并输出标准的 Tau, Accuracy 和 PMR 指标。
    """
    if model is None:
        model = default_critic_model_sind() if sind else default_critic_model_rocs()
    
    model.to(DEVICE)
    model.eval()
    
    # 1. 读取对应的验证集数据
    val_paragraphs = sind_paragraphs('val') if sind else rocs.dataset_get()['val']
    
    total_tau = 0.0
    total_acc = 0.0
    total_pmr = 0.0
    total_count = 0
    
    for tgt_paragraph in tqdm(val_paragraphs, desc="Evaluating Sentence Ordering"):
        # 过滤掉少于2个句子的极度特殊段落（无法计算相关性）
        if len(tgt_paragraph) <= 1:
            continue
            
        # 2. 模拟测试环境：彻底打乱段落，并获取其真实的绝对位置标签
        # true_labels 形式如: [2.0, 0.0, 1.0]
        shuffled_paragraph, true_labels = create_shuffled_paragraph_with_labels(tgt_paragraph)
        
        # 3. 构造 BERT 输入并让模型预测分数
        input_ids, attention_mask = build_bert_input(shuffled_paragraph)
        input_ids_t = torch.tensor(input_ids).unsqueeze(0).to(DEVICE)
        attention_mask_t = torch.tensor(attention_mask).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            scores = model(input_ids_t, attention_mask_t) # 输出大小为 [num_sentences]
            
        # 4. 【核心步骤】将分数转化为预测的位置标签 (predicted_labels)
        # 因为 ListMLE 训练出高分在前（即位置0），低分在后
        # 我们用两次 argsort(descending=True) 即可直接把连续的分数转化为 0, 1, 2... 的排名
        scores_cpu = scores.cpu()
        predicted_labels = torch.argsort(torch.argsort(scores_cpu, descending=True)).tolist()
        
        # 5. 调用你提供的方法计算三大指标
        tau = cal_tau(predicted_labels, true_labels)
        acc = cal_acc(predicted_labels, true_labels)
        pmr = cal_PMR(predicted_labels, true_labels)
        
        # 防止因极端情况 kendalltau 返回 nan
        if np.isnan(tau):
            tau = 0.0
            
        total_tau += tau
        total_acc += acc
        total_pmr += pmr
        total_count += 1
        
    # 6. 计算全数据集的平均分
    mean_tau = total_tau / total_count if total_count > 0 else 0
    mean_acc = total_acc / total_count if total_count > 0 else 0
    mean_pmr = total_pmr / total_count if total_count > 0 else 0
    
    # 7. 打印漂亮的测试报告
    print(f"\n=================== Sentence Ordering Test Report ===================")
    print(f"Dataset 类型: {'SIND' if sind else 'ROCS'} | 总评估段落数: {total_count}")
    print(f"1. Kendall's Tau (τ)           : {mean_tau:.4f}  (越接近 1 越好)")
    print(f"2. Absolute Position Acc (Acc) : {mean_acc:.4f}  (位置完全正确的句子占比)")
    print(f"3. Perfect Match Rate (PMR)    : {mean_pmr:.4f}  (整篇文章完全排序正确的概率)")
    print(f"=====================================================================")
    
    return mean_tau, mean_acc, mean_pmr



def test_order_consistency(model=None, sind=True):
    """
    检验模型是否会因为输入顺序的变化而改变最终的排序结果。
    做法：对同一个段落随机打乱两次，分别预测，看恢复出来的最终顺序是否一致。
    """
    if model is None:
        model = default_critic_model_sind() if sind else default_critic_model_rocs()
    
    model.to(DEVICE)
    model.eval()
    
    val_paragraphs = sind_paragraphs('val') if sind else rocs.dataset_get()['val']
    
    inconsistent_count = 0
    total_count = 0
    
    for tgt_paragraph in tqdm(val_paragraphs, desc="Testing Order Consistency"):
        if len(tgt_paragraph) <= 1:
            continue
            
        # ================== 第一次打乱与预测 ==================
        # 带有原始索引的段落，例如：[(0, 'A'), (1, 'B'), (2, 'C')]
        indexed_p1 = list(enumerate(tgt_paragraph))
        random.shuffle(indexed_p1)
        shuffled_p1 = [item[1] for item in indexed_p1]
        orig_indices_1 = [item[0] for item in indexed_p1] # 记录当前位置对应的原始句子编号
        
        input_ids1, mask1 = build_bert_input(shuffled_p1)
        input_ids_t1 = torch.tensor(input_ids1).unsqueeze(0).to(DEVICE)
        mask_t1 = torch.tensor(mask1).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            scores1 = model(input_ids_t1, mask_t1).cpu().tolist()
            
        # 将“原始句子编号”和“模型预测得分”绑定，并按得分从大到小（降序）排序
        # 最终得到基于原始句子的复原顺序，例如 [0, 2, 1]
        resolved_order_1 = [orig_idx for score, orig_idx in sorted(zip(scores1, orig_indices_1), reverse=True)]
        # ================== 第二次打乱与预测 ==================
        indexed_p2 = list(enumerate(tgt_paragraph))
        random.shuffle(indexed_p2)
        shuffled_p2 = [item[1] for item in indexed_p2]
        orig_indices_2 = [item[0] for item in indexed_p2]
        
        input_ids2, mask2 = build_bert_input(shuffled_p2)
        input_ids_t2 = torch.tensor(input_ids2).unsqueeze(0).to(DEVICE)
        mask_t2 = torch.tensor(mask2).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            scores2 = model(input_ids_t2, mask_t2).cpu().tolist()
            
        # 同样的方法，还原出第二次预测的原始句子顺序
        resolved_order_2 = [orig_idx for score, orig_idx in sorted(zip(scores2, orig_indices_2), reverse=True)]
        # ================== 对比一致性 ==================
        # 只要最终还原出来的句子顺序有任何一个位置不同，就视作不一致
        if resolved_order_1 != resolved_order_2:
            inconsistent_count += 1
            
        total_count += 1
        
    # 计算概率
    inconsistency_rate = inconsistent_count / total_count if total_count > 0 else 0
    consistency_rate = 1.0 - inconsistency_rate
    
    print(f"\n=================== Robustness Test Report ===================")
    print(f"Dataset 类型: {'SIND' if sind else 'ROCS'} | 总评估段落数: {total_count}")
    print(f"结果不一致的段落数 (Inconsistent Pairs) : {inconsistent_count}")
    print(f"❌ 顺序不一致发生概率 (Inconsistency Rate): {inconsistency_rate:.4%}")
    print(f"✅ 模型决策一致性概率 (Consistency Rate)  : {consistency_rate:.4%}")
    print(f"==============================================================")
    
    return inconsistency_rate


def test_trained(sind=True, split='test'):
    """
    自动扫描 checkpoints 文件夹，加载所有训练好的 BERT4SO 模型，
    并在指定的数据集划分（默认 test 集）上跑全量指标测试。
    """
    from pathlib import Path
    directory_path = Path("./checkpoints")
    
    # 根据前面设定的命名规则，动态组合搜索字符串
    # 例如：'critic_bert_sind' 或 'critic_bert_rocs'
    dataset_tag = 'sind' if sind else 'rocs'
    search_string = f"listmle_rep"
    
    # 找出文件夹下所有匹配的 pth/ckpt 文件
    matching_files = [file for file in directory_path.glob(f"*{search_string}*") if file.is_file()]
    
    if not matching_files:
        print(f"❌ 未在 {directory_path} 中找到包含 '{search_string}' 的模型权重文件。")
        return

    # 加载对应的测试/验证数据
    test_paragraphs = sind_paragraphs(split) if sind else rocs.dataset_get()[split]
    print(f"🔍 找到 {len(matching_files)} 个匹配的模型，开始在 {dataset_tag.upper()} 的 【{split}】 集上进行测试...")

    for file in matching_files:
        # 1. 必须实例化完整的神经网络架构（包含 BERT 和 Linear Head）
        model = CriticBert()
        load_checkpoint(model, str(file))
        model.to(DEVICE)
        model.eval()
        
        total_tau = 0.0
        total_acc = 0.0
        total_pmr = 0.0
        total_count = 0
        
        # 2. 遍历测试集进行微观打分与排序还原
        for tgt_paragraph in test_paragraphs:
            if len(tgt_paragraph) <= 1:
                continue
                
            # 测试时同样需要随机打乱，看模型能否完美复原
            shuffled_paragraph, true_labels = create_shuffled_paragraph_with_labels(tgt_paragraph)
            
            input_ids, attention_mask = build_bert_input(shuffled_paragraph)
            input_ids_t = torch.tensor(input_ids).unsqueeze(0).to(DEVICE)
            attention_mask_t = torch.tensor(attention_mask).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                scores = model(input_ids_t, attention_mask_t)
                
            # 将输出的分数转换为绝对位置排名
            scores_cpu = scores.cpu()
            predicted_labels = torch.argsort(torch.argsort(scores_cpu, descending=True)).tolist()
            
            # 3. 调用你的算分函数
            tau = cal_tau(predicted_labels, true_labels)
            acc = cal_acc(predicted_labels, true_labels)
            pmr = cal_PMR(predicted_labels, true_labels)
            
            if np.isnan(tau):
                tau = 0.0
                
            total_tau += tau
            total_acc += acc
            total_pmr += pmr
            total_count += 1
            
        # 4. 计算当前模型的平均指标
        mean_tau = total_tau / total_count if total_count > 0 else 0
        mean_acc = total_acc / total_count if total_count > 0 else 0
        mean_pmr = total_pmr / total_count if total_count > 0 else 0
        
        result_str = f"Tau: {mean_tau:.4f} | Acc: {mean_acc:.4f} | PMR: {mean_pmr:.4f}"
        
        # 5. 打印并记录日志
        print(f'Model: {file.name} -> {result_str}')
        common.logging.warning(f'Model: {file.name} -> {result_str}')
        
    print("\n✅ 所有模型测试完毕！")