# SIND数据集
# train: 40155
# val: 4990
# test 5055
import json
from pydash.arrays import chunk
from bert_utils import default_tokenizer, BertInput, indexs_tokenized, default_bert, DEVICE, reverse_indexs_tokenized
import random
from scipy.stats import kendalltau
from tqdm import tqdm
import numpy as np

MAX_TOKENS = 512
NEED_PAD = True

def calculate_dataset_length_SIND():  
    prefixs = ['train', 'val', 'test']
    story_count = 0
    for prefix in prefixs:
        with open(f'/home/zhuobinggang/Downloads/SIS-with-labels/sis/{prefix}.story-in-sequence.json' , 'r', encoding='utf-8') as file:
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
    with open(f'/home/zhuobinggang/Downloads/SIS-with-labels/sis/{split}.story-in-sequence.json' , 'r', encoding='utf-8') as file:
        data = json.load(file)
    for item in data['annotations']:
        only_texts.append(item[0]['original_text'])
    return chunk(only_texts, 5)

def sind_encode_paragraph(sentence_idss, sentence_prefix_ids, paragraph_prefx_ids, paragraph_suffix_ids):
    # 加上前缀
    sentence_idss = [sentence_prefix_ids + sentence_ids for sentence_ids in sentence_idss]
    # 将每个故事的5个句子拼接成一个段落，加入CLS和SEP
    paragraph_ids = []
    for sentence_ids in sentence_idss:
        paragraph_ids += sentence_ids
    paragraph_ids = paragraph_prefx_ids + paragraph_ids + paragraph_suffix_ids
    return paragraph_ids

def default_sentence_prefix():
    toker = default_tokenizer()
    sentence_prefix = f'({toker.mask_token}) ' # keep the space at the end
    sentence_prefix_ids = toker.encode(sentence_prefix, add_special_tokens=False)
    return sentence_prefix_ids

def default_paragraph_prefix_and_suffix():
    toker = default_tokenizer()
    paragraph_prefx = f'{toker.cls_token} ' # keep the space at the end
    paragraph_prefx_ids = toker.encode(paragraph_prefx, add_special_tokens=False)
    paragraph_suffix = f' {toker.sep_token}' # keep the space at the beginning
    paragraph_suffix_ids = toker.encode(paragraph_suffix, add_special_tokens=False)
    return paragraph_prefx_ids, paragraph_suffix_ids

def sind_data_prepare(paragraphs = None):
    results = []
    # 1. 将json文件中的所有annotations
    if paragraphs is None:
        paragraphs = sind_only_texts_get_by_split('val')
    # 最大句子长度，超过这个长度的句子将被截断
    MAX_SENTENCE_IDS = 96 
    # 用tokenizer进行编码
    toker = default_tokenizer()
    # 句子前缀
    sentence_prefix_ids = default_sentence_prefix()
    # 段落前缀和后缀
    paragraph_prefx_ids, paragraph_suffix_ids = default_paragraph_prefix_and_suffix()
    # MASK TOKEN的ID
    mask_token_id = toker.mask_token_id
    for paragraph in paragraphs:
        assert len(paragraph) == 5, "Each story should have 5 sentences"
        # 打乱句子和标签
        indexs = list(range(len(paragraph)))
        index_sentence_pairs = list(zip(indexs, paragraph))
        random.shuffle(index_sentence_pairs)
        indexs, paragraph = zip(*index_sentence_pairs)
        labels = [indexs_tokenized()[index] for index in indexs] # 将标签转换为token id
        # 编码句子
        sentence_idss = toker.encode(paragraph, add_special_tokens=False)
        # trim sentence_ids
        sentence_idss = [sentence_ids[:MAX_SENTENCE_IDS] for sentence_ids in sentence_idss]
        # 将每个故事的5个句子拼接成一个段落，加入CLS和SEP
        token_ids = sind_encode_paragraph(sentence_idss, sentence_prefix_ids, paragraph_prefx_ids, paragraph_suffix_ids)
        # 准备label_ids
        label_ids = [-100] * len(token_ids) # -100 will be ignored in loss calculation
        counter = 0
        for idx, token_id in enumerate(token_ids):
            if token_id == mask_token_id:
                label_ids[idx] = labels[counter]
                counter += 1
        assert counter == 5, "There should be exactly 5 MASK tokens in the paragraph"
        # 准备attention_mask
        attention_mask = [1] * len(token_ids)
        # pad到最大长度
        if NEED_PAD:
            extra_length = MAX_TOKENS - len(token_ids)
            token_ids = token_ids + [toker.pad_token_id] * extra_length
            label_ids = label_ids + [-100] * extra_length
            attention_mask = attention_mask + [0] * extra_length
        results.append(BertInput(input_ids=token_ids, attention_mask=attention_mask, labels=label_ids))
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

def decode_by_default_model(input_ids):
    import torch
    toker = default_tokenizer()
    bert = default_bert()
    input_ids = torch.tensor([input_ids]).to(DEVICE)
    with torch.no_grad():
        logits = bert(input_ids = input_ids).logits # [1, 512, 30522]
    mask_token_index = (input_ids == toker.mask_token_id)[0].nonzero(as_tuple=True)[0]
    predicted_token_ids = logits[0, mask_token_index].argmax(axis=-1)
    assert len(predicted_token_ids) == 5, "There should be exactly 5 predicted token ids"
    return predicted_token_ids.tolist()

def cal_tau(predicted_labels, true_labels):
    tau, _ = kendalltau(predicted_labels, true_labels)
    return tau

def cal_acc(predicted_labels, true_labels):
    correct_count = sum(p == t for p, t in zip(predicted_labels, true_labels))
    return correct_count / len(true_labels)

def fix_predicted_sequence(pred):
    """
    将包含重复序号的非法序列，转换为合法的无重复置换序列。
    原理：保持原有的相对大小趋势，相同大小的按先后顺序排列（Stable Sort）。
    """
    pred = np.array(pred)
    # argsort 的两次调用是推荐的获取 Rank 且不重复的标准做法
    fixed_sequence = np.argsort(np.argsort(pred))
    return fixed_sequence.tolist()

def cal_tau_acc(all_predicted_labels, all_true_labels, need_fix = False):
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
    print(f"Average accuracy (after fixing): {avg_acc}")


# @acc = 0.212
# @tau = nan
# 原因是出现重复的预测标签，导致无法计算kendall tau
# 对标签进行修正后重新计算
# @acc = 0.202
# @tau = 0.022
def default_bert_decode_acc(split):
    # 首先将val数据集转换成BertInput格式
    # paragraphs = sind_only_texts_get_by_split('val')[:100] # 取前100个故事进行测试
    paragraphs = sind_only_texts_get_by_split(split)
    bert_inputs = sind_data_prepare(paragraphs)
    # 然后使用默认的BERT模型进行解码，计算准确率和tau值
    all_predicted_labels = []
    all_true_labels = []
    for bert_input in tqdm(bert_inputs):
        predicted_labels = decode_by_default_model(bert_input.input_ids)
        true_labels = [label for label in bert_input.labels if label != -100]
        assert len(predicted_labels) == len(true_labels) == 5, "There should be exactly 5 predicted and true labels"
        predicted_labels = [reverse_indexs_tokenized()[a] for a in predicted_labels]
        true_labels = [reverse_indexs_tokenized()[b] for b in true_labels]
        all_predicted_labels.append(predicted_labels)
        all_true_labels.append(true_labels)
    # 在修正标签之前计算一次
    cal_tau_acc(all_predicted_labels, all_true_labels, need_fix = False)
    print("Now fixing predicted labels and recalculating metrics...")
    cal_tau_acc(all_predicted_labels, all_true_labels, need_fix = True)
    return all_predicted_labels, all_true_labels


def calculate_random_baseline(split):
    paragraphs = sind_only_texts_get_by_split(split)
    all_true_labels = []
    all_predicted_labels = []
    for paragraph in paragraphs:
        indexs = list(range(len(paragraph)))
        random.shuffle(indexs)
        all_true_labels.append(indexs)
        predicts = list(range(len(paragraph)))
        random.shuffle(predicts)
        all_predicted_labels.append(predicts)
    cal_tau_acc(all_predicted_labels, all_true_labels, need_fix = False)
 