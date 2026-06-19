# 用bert来判断两个段落哪个更好
from two_pass_plus import *

def run():
    bert = default_bert()
    load_checkpoint(bert, './checkpoints/SIND_best_e1.pth' )
    # valid_bert(bert, 'test')
    paragraphs = sind_only_texts_get_by_split('train')
    train_inputs = sind_data_prepare(paragraphs)
    result = valid_bert_two_pass_plus(bert=bert, bert_inputs=train_inputs) # 这个要10分钟
    # 按照result['all_true_labels']的顺序对段落进行排序
    original_order_paragraphs = []
    for paragraph, true_label in zip(paragraphs, result['all_true_labels']):
        ordered_paragraph = [None] * 5
        for idx, label in enumerate(true_label):
            ordered_paragraph[idx] = paragraph[label - 1] # label是1-5的索引
        original_order_paragraphs.append(ordered_paragraph)
    # 将original_order_paragraphs和对应的all_true_labels, all_predicted_labels_first, all_predicted_labels_second一起保存到文件中
    import json
    output_data = []
    for paragraph, true_label, pred_first, pred_second in zip(original_order_paragraphs, result['all_true_labels'], result['all_predicted_labels_first'], result['all_predicted_labels_second']):
        if hasattr(pred_first[0], 'tolist'):
            pred_first = [label.tolist() for label in pred_first]
        if hasattr(pred_second[0], 'tolist'):
            pred_second = [label.tolist() for label in pred_second]
        output_data.append({
            'paragraph': paragraph,
            'true_label': true_label,
            'predicted_label_first': pred_first,
            'predicted_label_second': pred_second
        })
    with open('./temp_datasets/train_two_pass_results.json', 'w') as f:
        json.dump(output_data, f, indent=4)
    
        