from sind import *

def run():
    bert = default_bert()
    load_checkpoint(bert, './checkpoints/SIND_best_e1.pth' )
    # valid_bert(bert, 'test')
    valid_bert_batched(bert, 'test')

def test_all_in_folder():
    import os
    checkpoint_folder = './checkpoints'
    for filename in os.listdir(checkpoint_folder):
        if filename.endswith('.pth'):
            print(f"Testing checkpoint: {filename}")
            bert = default_bert()
            load_checkpoint(bert, os.path.join(checkpoint_folder, filename))
            valid_bert_batched(bert, 'val')