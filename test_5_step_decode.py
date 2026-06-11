from multi_step_decode import *

if __name__ == '__main__':
    bert = default_bert()
    load_checkpoint(bert, './checkpoints/SIND_best_e1.pth')
    bert.eval()
    
    print("=" * 60)
    print("Testing 5-Step Decoding")
    print("=" * 60)
    
    preds, trues = valid_bert_n_steps(bert, 'val', n_steps=5)
    print(cal_tau_acc_pmr(preds, trues))
    preds, trues = valid_bert_n_steps(bert, 'test', n_steps=5)
    print(cal_tau_acc_pmr(preds, trues))
