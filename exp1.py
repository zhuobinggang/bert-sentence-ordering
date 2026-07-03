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
def direct_mlm_with_correct_paragraphs(use_sind = True, split = 'val'):
    from test import test_trained_simple
    test_trained_simple(use_sind = use_sind, split = split, random_shuffle = False)


def bert4so_with_correct_paragraphs(use_sind = True, split = 'val'):
    import bert4so
    bert4so.test_trained(sind = use_sind, split = split, need_shuffle = False)