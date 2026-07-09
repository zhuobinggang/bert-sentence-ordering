# nips数据集: https://github.com/XMUDeepLIT/NSEG/tree/master/nips
# DONE: 需要处理nips数据，每一行都是一个abstract，里面有很多句子，句子之间以<eos>分隔。需要将每一行的abstract拆分成句子
# DONE: 需要统计nips的最大最小句子数，平均句子数，最大最小句子长度，平均句子长度等信息
# 路径是'lite_dataset/nips/{split}.lower'，其中split可以是'train'、'val'、'test'

import os
from functools import lru_cache

@lru_cache(maxsize=3)
def get_paragraphs(split, base_path='lite_dataset/nips/'):
    """
    读取指定 split 的 NIPS 数据集，并将每一行的 abstract 拆分成句子
    """
    file_path = os.path.join(base_path, f'{split}.lower')
    
    if not os.path.exists(file_path):
        print(f"⚠️ 文件不存在: {file_path}，已跳过该 split。")
        return []
    
    processed_data = []  # 存储处理后的句子列表，结构为：[[sentence1, sentence2, ...], [...]]
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # 跳过空行
            
            # 将每一行的 abstract 拆分成句子
            sentences = [s.strip() for s in line.split('<eos>') if s.strip()]
            
            if sentences:
                processed_data.append(sentences)
                
    return processed_data

def process_nips_dataset(split, base_path='lite_dataset/nips/'):
    """
    处理指定 split 的 NIPS 数据集，拆分句子并统计相关信息
    """
    file_path = os.path.join(base_path, f'{split}.lower')
    
    if not os.path.exists(file_path):
        print(f"⚠️ 文件不存在: {file_path}，已跳过该 split。")
        return None, None
    
    processed_data = []          # 存储处理后的句子列表，结构为：[[sentence1, sentence2, ...], [...]]
    num_sentences_per_abstract = []  # 记录每个摘要的句子数
    all_sentence_lengths = []    # 记录所有句子的长度（单词数）
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # 跳过空行
            
            # 1. 将每一行的 abstract 拆分成句子
            # 使用 <eos> 分隔，并去除每个句子首尾的空格，同时过滤掉因末尾有 <eos> 产生的空字符串
            sentences = [s.strip() for s in line.split('<eos>') if s.strip()]
            
            if not sentences:
                continue
                
            # 保存拆分后的句子
            processed_data.append(sentences)
            
            # 2. 收集统计信息
            # 记录当前摘要的句子数
            num_sentences_per_abstract.append(len(sentences))
            
            # 记录每个句子的长度（以空格分隔的单词数/Token数）
            for sentence in sentences:
                words = sentence.split()
                all_sentence_lengths.append(len(words))
                
    if not num_sentences_per_abstract:
        print(f"❌ 数据集 {split} 中未解析出有效数据。")
        return None, None
        
    # 3. 计算各项统计指标
    stats = {
        'split': split,
        'total_abstracts': len(num_sentences_per_abstract),
        'max_sentences': max(num_sentences_per_abstract),
        'min_sentences': min(num_sentences_per_abstract),
        'avg_sentences': sum(num_sentences_per_abstract) / len(num_sentences_per_abstract),
        'max_sentence_len': max(all_sentence_lengths),
        'min_sentence_len': min(all_sentence_lengths),
        'avg_sentence_len': sum(all_sentence_lengths) / len(all_sentence_lengths)
    }
    
    return processed_data, stats

def print_statistics(stats):
    """
    美化打印统计结果
    """
    if not stats:
        return
    print(f"======== {stats['split'].upper()} 数据集统计结果 ========")
    print(f"总摘要数 (Abstracts): {stats['total_abstracts']}")
    print(f"每个摘要的最大句子数: {stats['max_sentences']}")
    print(f"每个摘要的最小句子数: {stats['min_sentences']}")
    print(f"每个摘要的平均句子数: {stats['avg_sentences']:.2f}")
    print(f"最大句子长度 (单词数): {stats['max_sentence_len']}")
    print(f"最小句子长度 (单词数): {stats['min_sentence_len']}")
    print(f"平均句子长度 (单词数): {stats['avg_sentence_len']:.2f}")
    print("=" * 40 + "\n")

if __name__ == '__main__':
    # 设定数据集根目录
    DATASET_DIR = 'lite_dataset/nips/'
    splits = ['train', 'val', 'test']
    
    # 存储所有拆分后的数据，如果后续需要用到可以从这里获取
    all_processed_datasets = {}
    
    for split in splits:
        sentences_list, stats = process_nips_dataset(split, base_path=DATASET_DIR)
        if stats:
            print_statistics(stats)
            all_processed_datasets[split] = sentences_list
            
            # 【可选】如果你需要把拆分后的句子保存到新文件中，可以取消下方代码的注释：
            # save_path = os.path.join(DATASET_DIR, f'{split}_sentences.txt')
            # with open(save_path, 'w', encoding='utf-8') as sf:
            #     for abstract in sentences_list:
            #         for sentence in abstract:
            #             sf.write(sentence + '\n')
            #         sf.write('\n') # 每个abstract之间用空行隔开