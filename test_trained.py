from reader import *

bert = default_bert()
load_checkpoint(bert, '/home/zhuobinggang/research/bert-sentence-ordering/checkpoints/SIND_best_e0.pth' )
# valid_bert(bert, 'test')
valid_bert_batched(bert, 'test')