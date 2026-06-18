from rocs import train_dataloader_provider, val_dataloader_provider, dataset_get
from three_pass_random import valid_bert_three_pass_random
from aux_loss import AuxLossBert
from sind import load_checkpoint, DEVICE, train, sind_data_prepare
import common

def train_aux_loss_bert_rocs(suffix = '_aux_loss_rocs_bert'):
    model = AuxLossBert()
    model.to(DEVICE)
    train(epochs=5, 
          model=model, 
          suffix=suffix, 
          trian_dataloader_provider=train_dataloader_provider, 
          val_dataloader_provider=val_dataloader_provider)

def train_n_repeats(n = 5):
    for i in range(n):
        train_aux_loss_bert_rocs(suffix = f'_aux_loss_rocs_repeat_{i+1}')

def test():
    from pathlib import Path
    directory_path = Path("./checkpoints")
    search_string = '_aux_loss_rocs_repeat'
    matching_files = [file for file in directory_path.glob(f"*{search_string}*") if file.is_file()]
    test_set_bert_inputs = sind_data_prepare(dataset_get()[2])
    for file in matching_files:
        model = AuxLossBert()
        # the_path = the_path or './checkpoints/SIND_best_20260616_132444_815731pair_loss_bert_best_acc.pth'
        load_checkpoint(model, str(file))
        model.to(DEVICE)
        model.eval()
        result = valid_bert_three_pass_random(model.bert, split='', bert_inputs = test_set_bert_inputs)
        print(f'Model ROCS: {file}, Test Result: {result}')
        common.logging.warning(f'Model ROCS: {file}, Test Result: {result}')

if __name__ == "__main__":
    train_n_repeats(5)
    test()