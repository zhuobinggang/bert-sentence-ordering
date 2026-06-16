# SIND数据集
# train: 40155
# val: 4990
# test 5055
from functools import lru_cache
import json
from pydash.arrays import chunk
from bert_utils import default_tokenizer, BertInput, indexs_tokenized, default_bert, DEVICE, reverse_indexs_tokenized
import random
from scipy.stats import kendalltau
from tqdm import tqdm
import numpy as np
import common
import os
import torch
from torch import optim
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from recordclass import recordclass
import numpy as np
from scipy.optimize import linear_sum_assignment

TestResult = recordclass('TestResult', 'tau acc pmr')

MAX_TOKENS = 512
NEED_PAD = True
BATCH_SIZE = 4 if common.LOCAL else 32

def calculate_dataset_length_SIND():  
    prefixs = ['train', 'val', 'test']
    story_count = 0
    for prefix in prefixs:
        the_path = os.path.join(common.dataset_base, f'SIND/{prefix}.story-in-sequence.json')
        with open(the_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            split_story_count = len(data['annotations']) / 5 # 每个故事包含5个句子
            print(prefix, split_story_count)
            story_count += split_story_count
    print("Total stories:", story_count)
    assert story_count == 50200, "Total story count should be 50200"

def calculate_dataset_length_ROCS():
    from datasets import load_dataset
    ds = load_dataset("mintujupally/ROCStories")
    assert len(ds['train']) == 78528, "Train set should have 78528 stories"
    assert len(ds['test']) == 19633, "Test set should have 19633 stories"


# @Response[0] for split=val
# ['My sister arrived early to help me with the family Bar BQ.', 
# 'Every one else arrived soon after.', 
# 'Dad manned the grill.', 
# 'There was so much food and it was all delicious.', 
# 'We ended the day shooting off some fireworks.']
def sind_only_texts_get_by_split(split):
    only_texts = []
    the_path = os.path.join(common.dataset_base, f'SIND/{split}.story-in-sequence.json')
    print(f"Loading SIND dataset from {the_path}...")
    with open(the_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    for item in data['annotations']:
        only_texts.append(item[0]['original_text'])
    return chunk(only_texts, 5)

# @sentence_index: 句子的索引标记（从1开始），如果不提供则使用MASK token
@lru_cache(maxsize=10)
def default_sentence_prefix(sentence_index = None):
    toker = default_tokenizer()
    if sentence_index is None:
        sentence_index = toker.mask_token
    sentence_prefix = f'({sentence_index}) ' # keep the space at the end
    sentence_prefix_ids = toker.encode(sentence_prefix, add_special_tokens=False)
    return sentence_prefix_ids

@lru_cache(maxsize=1)
def default_paragraph_prefix_and_suffix():
    common.print_once("不使用Sentence ordering:")
    toker = default_tokenizer()
    paragraph_prefx = f'{toker.cls_token} ' # keep the space at the end
    paragraph_prefx_ids = toker.encode(paragraph_prefx, add_special_tokens=False)
    paragraph_suffix = f' {toker.sep_token}' # keep the space at the beginning
    paragraph_suffix_ids = toker.encode(paragraph_suffix, add_special_tokens=False)
    return paragraph_prefx_ids, paragraph_suffix_ids

@lru_cache(maxsize=1)
def paragraph_prefix_suffix_with_instruct():
    common.print_once("使用Sentence ordering:")
    toker = default_tokenizer()
    paragraph_prefx = f'{toker.cls_token} Sentence ordering: ' # keep the space at the end
    paragraph_prefx_ids = toker.encode(paragraph_prefx, add_special_tokens=False)
    paragraph_suffix = f' {toker.sep_token}' # keep the space at the beginning
    paragraph_suffix_ids = toker.encode(paragraph_suffix, add_special_tokens=False)
    return paragraph_prefx_ids, paragraph_suffix_ids

def add_one(lst):
    return [x + 1 for x in lst]

def shuffle_paragraph(paragraph, need_add_one = True, need_shuffle = True):
    indexs = list(range(len(paragraph)))
    # NOTE: 标签从1开始
    if need_add_one:
        indexs = add_one(indexs)
    index_sentence_pairs = list(zip(indexs, paragraph))
    if need_shuffle:
        random.shuffle(index_sentence_pairs)
    indexs, paragraph = zip(*index_sentence_pairs)
    return list(indexs), list(paragraph)

def default_prefix_suffix_provider():
    sentence_prefix_ids = default_sentence_prefix()
    paragraph_prefx_ids, paragraph_suffix_ids = default_paragraph_prefix_and_suffix()
    return sentence_prefix_ids, paragraph_prefx_ids, paragraph_suffix_ids

def sentence_prefix_random_mask(indexs, random_mask_count = 5, output_mask_indices = False):
    random_mask_count = min(random_mask_count, len(indexs))
    random_mask_indices = random.sample(range(len(indexs)), random_mask_count) # 随机选择n个句子进行MASK
    random_mask_indices = sorted(random_mask_indices) # 将随机选择的索引排序，保证顺序不变
    sentence_prefix_ids_by_sentence = []
    for i, sentence_index in enumerate(indexs):
        if i in random_mask_indices:
            sentence_prefix_ids_by_sentence.append(default_sentence_prefix()) # 使用MASK token作为前缀
        else:
            sentence_prefix_ids_by_sentence.append(default_sentence_prefix(sentence_index))
    if output_mask_indices:
        return sentence_prefix_ids_by_sentence, random_mask_indices
    else:
        return sentence_prefix_ids_by_sentence

# Output:
# ['( 1 )', '( 2 )', '( 3 )', '( 4 )', '( 5 )']
# ['( 1 )', '( 2 )', '( 3 )', '( [MASK] )', '( 5 )']
# ['( [MASK] )', '( 2 )', '( 3 )', '( [MASK] )', '( 5 )']
# ['( [MASK] )', '( [MASK] )', '( 3 )', '( [MASK] )', '( 5 )']
# ['( [MASK] )', '( 2 )', '( [MASK] )', '( [MASK] )', '( [MASK] )']
# ['( [MASK] )', '( [MASK] )', '( [MASK] )', '( [MASK] )', '( [MASK] )']
def test_sentence_prefix_random_mask():
    indexs = [1, 2, 3, 4, 5]
    toker = default_tokenizer()
    for random_mask_count in range(6):
        print(toker.decode(sentence_prefix_random_mask(indexs, random_mask_count=random_mask_count)))

# @paragraph: 打乱顺序后的段落文本列表
# @indexs: 打乱顺序后的标签列表 (从1开始，表示原来的第几句话)
def create_bert_input_for_shuffled_paragraph(paragraph, indexs, MAX_SENTENCE_IDS = 96, paragraph_prefix_suffix_provider = default_paragraph_prefix_and_suffix, random_mask_count = 5):
    toker = default_tokenizer()
    labels = [indexs_tokenized()[index] for index in indexs] # 将标签转换为token id
    # 编码句子
    sentence_idss = toker.encode(paragraph, add_special_tokens=False)
    # trim sentence_ids
    sentence_idss = [sentence_ids[:MAX_SENTENCE_IDS] for sentence_ids in sentence_idss]
    # 获取前缀和后缀
    paragraph_prefx_ids, paragraph_suffix_ids = paragraph_prefix_suffix_provider()
    # 将每个故事的5个句子拼接成一个段落，加入CLS和SEP
    # 句子前缀
    common.print_once(f"随机MASK{random_mask_count}个句子前缀!")
    sentence_prefix_ids_by_sentence, random_mask_indices = sentence_prefix_random_mask(indexs, random_mask_count=random_mask_count, output_mask_indices=True) # 随机选取若干句子进行MASK，默认是全部句子都进行MASK
    sentence_idss_with_prefix = []
    for prefix_ids, sentence_ids in zip(sentence_prefix_ids_by_sentence, sentence_idss):
        sentence_idss_with_prefix.append(prefix_ids + sentence_ids)
    # 段落前后缀
    token_ids = paragraph_prefx_ids + [a for sentence_ids in sentence_idss_with_prefix for a in sentence_ids] + paragraph_suffix_ids
    # 准备label_ids: 将MASK token位置的label_id设置为对应的标签，其余位置设置为-100（在计算loss时会被忽略）
    label_ids = [-100] * len(token_ids) # -100 will be ignored in loss calculation
    counter = 0
    for idx, token_id in enumerate(token_ids):
        if token_id == toker.mask_token_id:
            label_ids[idx] = labels[random_mask_indices[counter]] # 只有被MASK的句子才有标签
            counter += 1
    # 准备attention_mask: 基本上所有token都参与attention，除了padding部分
    attention_mask = [1] * len(token_ids)
    # pad到最大长度
    if NEED_PAD:
        extra_length = MAX_TOKENS - len(token_ids)
        token_ids = token_ids + [toker.pad_token_id] * extra_length
        label_ids = label_ids + [-100] * extra_length
        attention_mask = attention_mask + [0] * extra_length
    bert_input = BertInput(input_ids=token_ids, attention_mask=attention_mask, labels=label_ids)
    return bert_input

def sind_data_prepare(paragraphs, random_mask_count = 5, need_shuffle = True):
    results = []
    # 最大句子长度，超过这个长度的句子将被截断
    MAX_SENTENCE_IDS = 96 
    # 段落前缀和后缀
    paragraph_prefix_suffix_provider = paragraph_prefix_suffix_with_instruct if common.args.instruction else default_paragraph_prefix_and_suffix
    for paragraph in paragraphs:
        # assert len(paragraph) == 5, "Each story should have 5 sentences"
        # 打乱句子和标签
        indexs, paragraph = shuffle_paragraph(paragraph, need_add_one = True, need_shuffle = need_shuffle) # 标签从1开始
        # 生成BertInput
        bert_input = create_bert_input_for_shuffled_paragraph(paragraph, indexs, MAX_SENTENCE_IDS, paragraph_prefix_suffix_provider=paragraph_prefix_suffix_provider, random_mask_count=random_mask_count)
        results.append(bert_input)
    return results

def input_ids_for_test():
    paragraphs = sind_only_texts_get_by_split('val')[:1]
    results = sind_data_prepare(paragraphs)
    return results[0]

def test():
    paragraphs = sind_only_texts_get_by_split('val')[:5]
    results = sind_data_prepare(paragraphs)
    for item in results:
        print(item)

def decode_by_bert_keep_repeated(input_ids, attention_mask, bert=None):
    toker = default_tokenizer()
    if bert is None:
        bert = default_bert()
    input_ids = torch.tensor([input_ids]).to(DEVICE)
    attention_mask = torch.tensor([attention_mask]).to(DEVICE)
    with torch.no_grad():
        logits = bert(input_ids = input_ids, attention_mask = attention_mask).logits # [1, 512, 30522]
    mask_token_bool = (input_ids == toker.mask_token_id)
    predicted_token_ids = logits[mask_token_bool].argmax(axis=-1)  # [5]
    assert len(predicted_token_ids) == 5, "There should be exactly 5 predicted token ids"
    return predicted_token_ids.tolist()

# 使用匈牙利算法解码得到标签
def decode_by_bert(input_ids, attention_mask, bert=None):
    toker = default_tokenizer()
    if bert is None:
        bert = default_bert()
    input_ids = torch.tensor([input_ids]).to(DEVICE)
    attention_mask = torch.tensor([attention_mask]).to(DEVICE)
    with torch.no_grad():
        logits = bert(input_ids = input_ids, attention_mask = attention_mask).logits # [1, 512, 30522]
    mask_token_bool = (input_ids == toker.mask_token_id)
    predicted_token_ids = logits[mask_token_bool]  # [5, vocab_size]
    index_dict = indexs_tokenized()
    index_1_to_5_token_ids = [index_dict[i] for i in range(1, 6)]
    predicted_token_ids = predicted_token_ids[:, index_1_to_5_token_ids] # [5, 5] 每个mask位置对应5个标签的logits
    predicted_labels = get_valid_permutation(predicted_token_ids.cpu().numpy())
    assert len(predicted_labels) == 5, "There should be exactly 5 predicted token ids"
    return predicted_labels.tolist()

def cal_tau(predicted_labels, true_labels):
    tau, _ = kendalltau(predicted_labels, true_labels)
    return tau

def cal_acc(predicted_labels, true_labels):
    correct_count = sum(p == t for p, t in zip(predicted_labels, true_labels))
    return correct_count / len(true_labels)

def cal_PMR(predicted_labels, true_labels):
    for p, t in zip(predicted_labels, true_labels):
        if p != t:
            return 0
    return 1

def fix_predicted_sequence(pred):
    """
    将包含重复序号的非法序列，转换为合法的无重复置换序列。
    原理：保持原有的相对大小趋势，相同大小的按先后顺序排列（Stable Sort）。
    """
    pred = np.array(pred)
    # argsort 的两次调用是推荐的获取 Rank 且不重复的标准做法
    fixed_sequence = np.argsort(np.argsort(pred))
    return add_one(fixed_sequence.tolist())

def cal_tau_acc_pmr(all_predicted_labels, all_true_labels, need_fix = False):
    if need_fix:
        all_predicted_labels = [fix_predicted_sequence(pred) for pred in all_predicted_labels]
        all_true_labels = [fix_predicted_sequence(true) for true in all_true_labels]
    taus = []
    for predicted_labels, true_labels in zip(all_predicted_labels, all_true_labels):
        tau = cal_tau(predicted_labels, true_labels)
        taus.append(tau)
    avg_tau = sum(taus) / len(taus)
    print(f"Average Kendall's tau: {avg_tau}")
    accs = []
    for predicted_labels, true_labels in zip(all_predicted_labels, all_true_labels):
        acc = cal_acc(predicted_labels, true_labels)
        accs.append(acc)
    avg_acc = sum(accs) / len(accs)
    print(f"Average accuracy: {avg_acc}")
    pmrs = []
    for predicted_labels, true_labels in zip(all_predicted_labels, all_true_labels):
        pmr = cal_PMR(predicted_labels, true_labels)
        pmrs.append(pmr)
    avg_pmr = sum(pmrs) / len(pmrs)
    print(f"Average PMR: {avg_pmr}")
    return TestResult(tau=avg_tau, acc=avg_acc, pmr=avg_pmr)


def bert_inputs_to_dataloader_shuffle(bert_inputs):
    all_input_ids = torch.tensor([bert_input.input_ids for bert_input in bert_inputs], dtype=torch.long)
    all_attention_mask = torch.tensor([bert_input.attention_mask for bert_input in bert_inputs], dtype=torch.long)
    all_label_ids = torch.tensor([bert_input.labels for bert_input in bert_inputs], dtype=torch.long)
    datas = TensorDataset(all_input_ids, all_attention_mask, all_label_ids)
    sampler = RandomSampler(datas)
    dataloader = DataLoader(datas, sampler=sampler, batch_size=BATCH_SIZE)
    return dataloader

# 保留重复标签的版本，直接选取每个位置得分最高的标签，可能会有重复标签，导致tau计算出错
def valid_bert_keep_repeated(bert = None, split = 'val'):
    if bert is None:
        bert = default_bert()
    # 首先将val数据集转换成BertInput格式
    # paragraphs = sind_only_texts_get_by_split('val')[:100] # 取前100个故事进行测试
    paragraphs = sind_only_texts_get_by_split(split)
    bert_inputs = sind_data_prepare(paragraphs)
    # 然后使用默认的BERT模型进行解码，计算准确率和tau值
    all_predicted_labels = []
    all_true_labels = []
    reversed_dict = reverse_indexs_tokenized()
    for bert_input in tqdm(bert_inputs):
        predicted_labels = decode_by_bert_keep_repeated(bert_input.input_ids, bert_input.attention_mask, bert) # 注意要传递attention_mask
        true_labels = [label for label in bert_input.labels if label != -100]
        assert len(predicted_labels) == len(true_labels) == 5, "There should be exactly 5 predicted and true labels"
        predicted_labels = [reversed_dict.get(a, 5) for a in predicted_labels]
        true_labels = [reversed_dict[b] for b in true_labels]
        all_predicted_labels.append(predicted_labels)
        all_true_labels.append(true_labels)
    # 在修正标签之前计算一次
    test_result = cal_tau_acc_pmr(all_predicted_labels, all_true_labels, need_fix = False)
    # print("Now fixing predicted labels and recalculating metrics...")
    # _ = cal_tau_acc(all_predicted_labels, all_true_labels, need_fix = True)
    return test_result

# 保留重复标签的版本，直接选取每个位置得分最高的标签，可能会有重复标签，导致tau计算出错
def valid_bert_batched_keep_repeated(bert = None, split = 'val', split_length = None, dataloader = None):
    if bert is None:
        bert = default_bert()
    bert.eval()
    toker = default_tokenizer()
    # 首先将val数据集转换成BertInput格式
    # paragraphs = sind_only_texts_get_by_split('val')[:100] # 取前100个故事进行测试
    if dataloader is None:
        paragraphs = sind_only_texts_get_by_split(split)
        if split_length is not None:
            common.print_once(f"只使用{split}前{split_length}个故事进行验证")
            paragraphs = paragraphs[:split_length]
        bert_inputs = sind_data_prepare(paragraphs)
        dataloader = bert_inputs_to_dataloader_shuffle(bert_inputs)
    # 然后使用默认的BERT模型进行解码，计算准确率和tau值
    all_predicted_labels = []
    all_true_labels = []
    reversed_dict = reverse_indexs_tokenized()
    for batch in tqdm(dataloader, desc="Validation"):
        input_ids, attention_mask, label_ids = batch
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        with torch.no_grad():
            logits = bert(input_ids=input_ids, attention_mask=attention_mask).logits # # [batch_size, 512, 30522]
        # mask_token_index = (input_ids == toker.mask_token_id).nonzero(as_tuple=True)
        for i in range(input_ids.size(0)): # 遍历batch中的每个样本
            mask_token_bool = (input_ids[i] == toker.mask_token_id)
            predicted_token_ids = logits[i, mask_token_bool].argmax(axis=-1) # [5]
            true_label_ids = label_ids[i][label_ids[i] != -100] # [5]
            assert len(predicted_token_ids) == len(true_label_ids) == 5, "There should be exactly 5 predicted and true labels"
            predicted_labels = [reversed_dict.get(a.item(), 5) for a in predicted_token_ids]
            true_labels = [reversed_dict[b.item()] for b in true_label_ids]
            all_predicted_labels.append(predicted_labels)
            all_true_labels.append(true_labels)
    # 在修正标签之前计算一次
    test_result = cal_tau_acc_pmr(all_predicted_labels, all_true_labels, need_fix = False)
    # print("Now fixing predicted labels and recalculating metrics...")
    # _ = cal_tau_acc_pmr(all_predicted_labels, all_true_labels, need_fix = True)
    return test_result


# 匈牙利算法 2026.6.15
def get_valid_permutation(model_outputs):
    """
    model_outputs: 模型的原始输出。
    假设形状为 (5, 5)，即 5个句子，每个句子对应 5个位置的 logit 或 softmax 概率值。
    model_outputs[i][j] 表示第 i 个句子填入位置 j (0-4) 的得分。
    """
    # 1. 转换为 numpy 数组
    score_matrix = np.array(model_outputs)
    
    # 2. 因为 scipy 的 linear_sum_assignment 寻找的是完美匹配的“最小代价”
    # 我们要找的是“最大得分”，所以将矩阵取负号，将其转化为求最小值问题
    cost_matrix = -score_matrix
    
    # 3. 运行匈牙利算法
    # row_ind 会是 [0, 1, 2, 3, 4] （句子的索引）
    # col_ind 会是算法分配的、绝对不重复的 [p0, p1, p2, p3, p4] （位置的索引）
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # 4. col_ind 就是最终生成的无重复完美排列（0~4 映射）
    # 如果你的评估代码需要 1~5 的标签，直接 + 1 即可
    final_positions = col_ind + 1
    
    return final_positions


# 使用匈牙利算法解码得到标签
def valid_bert_batched(bert = None, split = 'val', split_length = None, dataloader = None):
    if bert is None:
        bert = default_bert()
    bert.eval()
    toker = default_tokenizer()
    # 首先将val数据集转换成BertInput格式
    # paragraphs = sind_only_texts_get_by_split('val')[:100] # 取前100个故事进行测试
    if dataloader is None:
        paragraphs = sind_only_texts_get_by_split(split)
        if split_length is not None:
            common.print_once(f"只使用{split}前{split_length}个故事进行验证")
            paragraphs = paragraphs[:split_length]
        bert_inputs = sind_data_prepare(paragraphs)
        dataloader = bert_inputs_to_dataloader_shuffle(bert_inputs)
    # 然后使用默认的BERT模型进行解码，计算准确率和tau值
    all_predicted_labels = []
    all_true_labels = []
    reversed_dict = reverse_indexs_tokenized()
    index_dict = indexs_tokenized()
    index_1_to_5_token_ids = [index_dict[i] for i in range(1, 6)]
    for batch in tqdm(dataloader, desc="Validation"):
        input_ids, attention_mask, label_ids = batch
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        with torch.no_grad():
            logits = bert(input_ids=input_ids, attention_mask=attention_mask).logits # # [batch_size, 512, 30522]
        # mask_token_index = (input_ids == toker.mask_token_id).nonzero(as_tuple=True)
        for i in range(input_ids.size(0)): # 遍历batch中的每个样本
            mask_token_bool = (input_ids[i] == toker.mask_token_id)
            # predicted_token_ids = logits[i, mask_token_bool].argmax(axis=-1) # [5]
            predicted_token_ids = logits[i, mask_token_bool] # [5, vocab_size]
            predicted_token_ids = predicted_token_ids[:, index_1_to_5_token_ids] # [5, 5] 每个mask位置对应5个标签的logits
            predicted_labels = get_valid_permutation(predicted_token_ids.cpu().numpy()) # [5] 每个位置的最终标签（1-5）
            true_label_ids = label_ids[i][label_ids[i] != -100] # [5]
            assert len(predicted_token_ids) == len(true_label_ids) == 5, "There should be exactly 5 predicted and true labels"
            # predicted_labels = [reversed_dict.get(a.item(), 5) for a in predicted_token_ids]
            true_labels = [reversed_dict[b.item()] for b in true_label_ids]
            all_predicted_labels.append(predicted_labels)
            all_true_labels.append(true_labels)
    # 在修正标签之前计算一次
    test_result = cal_tau_acc_pmr(all_predicted_labels, all_true_labels, need_fix = False)
    # print("Now fixing predicted labels and recalculating metrics...")
    # _ = cal_tau_acc_pmr(all_predicted_labels, all_true_labels, need_fix = True)
    return test_result

def valid_bert(bert = None, split = 'val'):
    if bert is None:
        bert = default_bert()
    # 首先将val数据集转换成BertInput格式
    # paragraphs = sind_only_texts_get_by_split('val')[:100] # 取前100个故事进行测试
    paragraphs = sind_only_texts_get_by_split(split)
    bert_inputs = sind_data_prepare(paragraphs)
    # 然后使用默认的BERT模型进行解码，计算准确率和tau值
    all_predicted_labels = []
    all_true_labels = []
    reversed_dict = reverse_indexs_tokenized()
    index_dict = indexs_tokenized()
    for bert_input in tqdm(bert_inputs):
        # NOTE: 这里使用匈牙利算法解码，得到无重复的标签序列，且直接就是1-5的标签，不需要再转换了
        predicted_labels = decode_by_bert(bert_input.input_ids, bert_input.attention_mask, bert) # 注意要传递attention_mask
        true_labels = [label for label in bert_input.labels if label != -100]
        true_labels = [reversed_dict[b] for b in true_labels]
        all_predicted_labels.append(predicted_labels)
        all_true_labels.append(true_labels)
    # 在修正标签之前计算一次
    test_result = cal_tau_acc_pmr(all_predicted_labels, all_true_labels, need_fix = False)
    # print("Now fixing predicted labels and recalculating metrics...")
    # _ = cal_tau_acc(all_predicted_labels, all_true_labels, need_fix = True)
    return test_result

def calculate_random_baseline(split):
    paragraphs = sind_only_texts_get_by_split(split)
    all_true_labels = []
    all_predicted_labels = []
    for paragraph in paragraphs:
        indexs = add_one(list(range(len(paragraph))))
        random.shuffle(indexs)
        all_true_labels.append(indexs)
        predicts = add_one(list(range(len(paragraph))))
        random.shuffle(predicts)
        all_predicted_labels.append(predicts)
    test_result = cal_tau_acc_pmr(all_predicted_labels, all_true_labels, need_fix = False)
    return test_result

def calculate_all_one_baseline(split):
    paragraphs = sind_only_texts_get_by_split(split)
    all_true_labels = []
    all_predicted_labels = []
    for paragraph in paragraphs:
        indexs = add_one(list(range(len(paragraph))))
        random.shuffle(indexs)
        all_true_labels.append(indexs)
        predicts = [1] * len(paragraph) # 全部预测为1
        all_predicted_labels.append(predicts)
    all_predicted_labels = [fix_predicted_sequence(pred) for pred in all_predicted_labels]
    test_result = cal_tau_acc_pmr(all_predicted_labels, all_true_labels, need_fix = False)
    return test_result


def save_checkpoint(bert, base_path = 'checkpoints', epoch = -1, valid_score = -1, suffix = ''):
    # path = f'{base_path}/{self.prefix}_epoch_{epoch}.pth'
    path = f'{base_path}/SIND_best_{suffix}.pth'
    torch.save({
        'epoch': epoch,
        'state': bert.state_dict(),
        'valid_score': valid_score,
    }, path)

def load_checkpoint(bert, path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    bert.load_state_dict(checkpoint['state'])
    bert.valid_score = checkpoint.get('valid_score', -1)
    bert.stop_epoch = checkpoint.get('epoch', -1)

def default_trian_dataloader_provider():
    print('重新制备训练数据集...')
    return bert_inputs_to_dataloader_shuffle(sind_data_prepare(sind_only_texts_get_by_split('train')))

@lru_cache(maxsize=1) # 只缓存验证数据集，训练数据集每次都重新制备，增加随机性
def default_val_dataloader_provider():
    print('重新制备验证数据集...')
    return bert_inputs_to_dataloader_shuffle(sind_data_prepare(sind_only_texts_get_by_split('val')))

def train(epochs = 5, suffix = '', 
          trian_dataloader_provider = default_trian_dataloader_provider, 
          val_dataloader_provider = default_val_dataloader_provider,
          model = None):
    # 准备valid数据集并固定
    val_dataloader = val_dataloader_provider()
    # 记录日志
    logger = common.logging.getLogger(__name__)
    writer = common.get_writer()
    # Train
    from accelerate import Accelerator
    accelerator = Accelerator()
    # model.cuda()
    if model is None:
        model = default_bert()
        model.train()
        optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    else:
        print('使用传入的模型进行训练...')
        assert hasattr(model, 'pair_classifier'), "传入的模型必须包含pair_classifier属性"
        optimizer_groups = [
            {"params": model.bert.parameters(), "lr": 5e-5},  
            {"params": model.pair_classifier.parameters(), "lr": 5e-3} 
        ]
        optimizer = optim.AdamW(optimizer_groups)
    model, optimizer = accelerator.prepare(
        model, optimizer
    )
    model_suffix = common.get_time_str() + suffix
    MAX_ACC = 0
    steps = 0
    for epoch in range(epochs): # 训练指定数量的epoch
        train_dataloader = accelerator.prepare(trian_dataloader_provider())
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            if batch_idx % 1000 == 0:
                logger.warning(f'{common.get_time_str()} Training iteration {batch_idx}')
            input_ids, attention_mask, label_ids = batch
            # NOTE: 2025.5.11 RoBERTa don't use token_type_ids! Error happens if use it!
            outputs = model(input_ids=input_ids.to(DEVICE), 
                    attention_mask=attention_mask.to(DEVICE),
                    labels=label_ids.to(DEVICE))
            loss = outputs.loss
            accelerator.backward(loss)
            writer.add_scalar(f'Loss', loss.item(), writer.global_step)
            if hasattr(model, 'pair_classifier'):
                writer.add_scalar(f'Pair_Loss', outputs.pair_loss, writer.global_step)
            writer.global_step += 1
            optimizer.step()
            optimizer.zero_grad()
            steps += 1
            if steps % 1000 == 0:
                model.eval()
                if hasattr(model, 'pair_classifier'):
                    common.print_once("aux模型, 使用model.bert进行验证")
                    score = valid_bert_batched(model.bert, dataloader=val_dataloader)
                else:
                    score = valid_bert_batched(model, dataloader=val_dataloader)
                model.train()
                print(f'{steps}检验模型，当前验证结果: {score}')
                if score.acc > MAX_ACC:
                    print('保存模型，当前准确率提升到{score.acc}，之前的最高准确率是{MAX_ACC}')
                    MAX_ACC = score.acc
                    save_checkpoint(model, base_path='checkpoints', epoch=epoch, valid_score=str(score), suffix=f'{model_suffix}_best_acc')
    model.eval()
    score = valid_bert_batched(model, dataloader=val_dataloader)
    model.train()
    print(f'最后一次检验模型，当前验证结果: {score}')
    if score.acc > MAX_ACC:
        print('保存模型，当前准确率提升到{score.acc}，之前的最高准确率是{MAX_ACC}')
        MAX_ACC = score.acc
        save_checkpoint(model, base_path='checkpoints', epoch=epoch, valid_score=str(score), suffix=f'{model_suffix}_best_acc')
    return model

