from common import print_only_once

def test_trained_simple(use_sind = True, split = 'val', random_shuffle = True):
    import sind
    import rocs
    from tqdm import tqdm
    import vanilla_sind
    import vanilla_rocs
    import bert_utils
    import common
    import random
    paragraphs = sind.sind_paragraphs(split) if use_sind else rocs.dataset_get()[split]
    checkpoint_paths = vanilla_sind.checkpoint_paths() if use_sind else vanilla_rocs.checkpoint_paths()
    taus = []
    accs = []
    pmrs = []
    for file in checkpoint_paths:
        bert = bert_utils.default_bert()
        sind.load_checkpoint(bert, str(file))
        bert.to(sind.DEVICE)
        bert.eval()
        temp_taus = []
        temp_accs = []
        temp_pmrs = []
        for paragraph in tqdm(paragraphs):
            true_labels = list(range(1, 6))
            if random_shuffle:
                msg = f'original paragraph: {paragraph}, true_labels: {true_labels}'
                sentence_label_pairs = list(zip(paragraph, true_labels))
                random.shuffle(sentence_label_pairs)
                paragraph, true_labels = zip(*sentence_label_pairs)
                msg += f'\nshuffled paragraph: {paragraph}, shuffled_labels: {true_labels}'
                print_only_once(msg)
            bert_input = sind.create_bert_input_for_shuffled_paragraph(paragraph, true_labels)
            temp_mask_token_5index_logits = sind.get_mask_token_5index_logits(bert_input.input_ids, bert_input.attention_mask, bert)
            temp_predicted_labels = sind.hungarian_algorithm_best_order(temp_mask_token_5index_logits.cpu().numpy())
            temp_taus.append(sind.cal_tau(temp_predicted_labels, true_labels))
            temp_accs.append(sind.cal_acc(temp_predicted_labels, true_labels))
            temp_pmrs.append(sind.cal_PMR(temp_predicted_labels, true_labels))
        taus.append(sum(temp_taus) / len(temp_taus))
        accs.append(sum(temp_accs) / len(temp_accs))
        pmrs.append(sum(temp_pmrs) / len(temp_pmrs))
    _ = common.cal_mean_std(taus)
    _ = common.cal_mean_std(accs)
    _ = common.cal_mean_std(pmrs)