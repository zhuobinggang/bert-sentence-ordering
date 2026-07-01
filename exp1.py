# 一些补充实验

# 计算模型在 coherency 上的表现，npass 次解码，如果每次解码的结果都一致，则认为是 coherent 的
# 结果已汇报
def two_pass_coherency():
    import critic_randomk
    # SIND
    critic_randomk.valid_bert_n_pass_coherency(sind = True, split = 'val', npass = 2)
    # ROCStory
    critic_randomk.valid_bert_n_pass_coherency(sind = False, split = 'val', npass = 2)

# 分析全部正序时候的Direct MLM的表现
def direct_mlm_with_correct_paragraphs(sind = True, split = 'val'):
    import sind
    import rocs
    from tqdm import tqdm
    import vanilla_sind
    import vanilla_rocs
    import bert_utils
    import common
    # 获取所有段落，不打乱
    paragraphs = sind.sind_paragraphs(split) if sind else rocs.dataset_get()[split]
    checkpoint_paths = vanilla_sind.checkpoint_paths() if sind else vanilla_rocs.checkpoint_paths()
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
