# 目前最好的代理： 使用aux loss，再加上打乱输入顺序三次，句子输出概率叠加
from three_pass_random import valid_bert_three_pass_random
from aux_loss import AuxLossBert, train_aux_loss_bert
from sind import load_checkpoint, DEVICE, sind_data_prepare, sind_only_texts_get_by_split, valid_bert_batched
import common

def train_n_repeats():
    n = common.args.repeats
    offset = common.args.offset
    for i in range(n):
        train_aux_loss_bert(suffix = f'_aux_loss_repeat_{i + offset}')

def test():
    from pathlib import Path
    directory_path = Path("./checkpoints")
    search_string = '_aux_loss_repeat'
    matching_files = [file for file in directory_path.glob(f"*{search_string}*") if file.is_file()]
    for file in matching_files:
        model = AuxLossBert()
        # the_path = the_path or './checkpoints/SIND_best_20260616_132444_815731pair_loss_bert_best_acc.pth'
        load_checkpoint(model, str(file))
        model.to(DEVICE)
        model.eval()
        result = valid_bert_three_pass_random(model.bert, 'test')
        print(f'Model: {file}, Test Result: {result}')
        common.logging.warning(f'Model: {file}, Test Result: {result}')

def test_one_pass():
    from pathlib import Path
    directory_path = Path("./checkpoints")
    search_string = '_aux_loss_repeat'
    matching_files = [file for file in directory_path.glob(f"*{search_string}*") if file.is_file()]
    dataloader = sind_data_prepare(sind_only_texts_get_by_split('test'))
    for file in matching_files:
        model = AuxLossBert()
        # the_path = the_path or './checkpoints/SIND_best_20260616_132444_815731pair_loss_bert_best_acc.pth'
        load_checkpoint(model, str(file))
        model.to(DEVICE)
        model.eval()
        result = valid_bert_batched(model.bert, dataloader=dataloader)
        print(f'Model: {file}, Test Result: {result}')
        common.logging.warning(f'Model: {file}, Test Result: {result}')


def test_pair_head():
    from pair_model import test_trained_by_pair_head
    from pathlib import Path
    directory_path = Path("./checkpoints")
    search_string = '_aux_loss_repeat'
    matching_files = [file for file in directory_path.glob(f"*{search_string}*") if file.is_file()]
    for file in matching_files:
        test_trained_by_pair_head(str(file))

if __name__ == "__main__":
    train_n_repeats()
    test()