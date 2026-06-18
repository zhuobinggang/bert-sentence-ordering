from rocs import *
from sind import DEVICE

def train_n_repeats(n = 5):
    for i in range(n):
        train(suffix=f'_vanilla_rocs_{i+1}', 
              trian_dataloader_provider=train_dataloader_provider, 
              val_dataloader_provider=val_dataloader_provider)
        
def test_trained():
    from pathlib import Path
    directory_path = Path("./checkpoints")
    search_string = '_vanilla_rocs_'
    matching_files = [file for file in directory_path.glob(f"*{search_string}*") if file.is_file()]
    test_dataloader = test_dataloader_provider()
    for file in matching_files:
        bert = default_bert()
        load_checkpoint(bert, str(file))
        bert.to(DEVICE)
        bert.eval()
        result = valid_bert_batched(bert, dataloader=test_dataloader)
        print(f'Model vanilla rocs: {file}, Test Result: {result}')
        common.logging.warning(f'Model vanilla rocs: {file}, Test Result: {result}')

if __name__ == "__main__":
    train_n_repeats(5)
    test_trained()