from sind import *
import rocs

# 计算sind数据集的信息
def sind_average_tokens():
    paragraphs = sind_paragraphs('val')
    toker = default_tokenizer()
    token_counts = []
    for paragraph in paragraphs:
        text = ' '.join(paragraph)
        tokens = toker.tokenize(text)
        token_counts.append(len(tokens))
    print(f'平均token数量: {sum(token_counts) / len(token_counts)}')

def rocs_average_tokens():
    paragraphs = rocs.dataset_get()['val']
    toker = default_tokenizer()
    token_counts = []
    for paragraph in paragraphs:
        text = ' '.join(paragraph)
        tokens = toker.tokenize(text)
        token_counts.append(len(tokens))
    print(f'平均token数量: {sum(token_counts) / len(token_counts)}')