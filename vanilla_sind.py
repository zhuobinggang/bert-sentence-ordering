from sind import train, default_bert, load_checkpoint, DEVICE, valid_bert_batched
import common

def checkpoint_paths():
    from pathlib import Path
    directory_path = Path("./checkpoints")
    search_string = '_vanilla_sind_'
    matching_files = [file for file in directory_path.glob(f"*{search_string}*") if file.is_file()]
    return matching_files


def train_n_repeats():
    n = common.args.repeats
    offset = common.args.offset
    for i in range(n):
        train(suffix = f'_vanilla_sind_{i + offset}')

def test_trained():
    ckpts = checkpoint_paths()
    for file in ckpts:
        bert = default_bert()
        load_checkpoint(bert, str(file))
        bert.to(DEVICE)
        bert.eval()
        result = valid_bert_batched(bert, 'test')
        print(f'Model vanilla sind: {file}, Test Result: {result}')
        common.logging.warning(f'Model vanilla sind: {file}, Test Result: {result}')

if __name__ == "__main__":
    train_n_repeats()
    test_trained()