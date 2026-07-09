# TODO: 需要重新实现bert_input函数以应对可变长输入
from sind import create_bert_input_for_shuffled_paragraph, MAX_TOKENS, shuffle_paragraph

def nips_bert_input(paragraph, need_shuffle = True):
    """
    将一个段落转换为BERT输入，段落是一个句子列表
    """
    sentences_length = len(paragraph)
    max_sentence_token_length = MAX_TOKENS // sentences_length if sentences_length > 0 else MAX_TOKENS
    max_sentence_token_length -= 3 # ([MASK]) 需要额外三个token
    random_mask_count = len(paragraph)  # 默认随机MASK所有句子
    labels, paragraph = shuffle_paragraph(paragraph, need_add_one = True, need_shuffle = need_shuffle) # 标签从1开始
    bert_input = create_bert_input_for_shuffled_paragraph(
        paragraph, 
        labels,
        random_mask_count=random_mask_count, 
        MAX_SENTENCE_IDS = max_sentence_token_length)
    return bert_input

def nips_data_prepare(paragraphs, need_shuffle = True):
    results = []
    for paragraph in paragraphs:
        # 生成BertInput
        bert_input = nips_bert_input(paragraph, need_shuffle=need_shuffle)
        results.append(bert_input)
    return results
