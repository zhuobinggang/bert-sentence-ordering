# 使用sentence-bert的拼装方法，然后加深分类头深度
from pair_model3 import *

DOUBLE_CHECK = False

class PairLossSentenceBERTDeep(PairLossSentenceBERT):
    def init_pair_classifier(self):
        # linear一个线性层接一个sigmoid层，1代表前后关系正确，0代表前后关系错误
        self.pair_classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )


def train_pair_loss_sentence_bert():
    model = PairLossSentenceBERTDeep()
    model.to(DEVICE)
    train(epochs=5, model=model, suffix='_pair_loss_sentence_bert_deep')

def test_trained():
    logger = common.logging.getLogger(__name__)
    from pathlib import Path
    directory_path = Path("./checkpoints")
    search_string = '_pair_loss_sentence_bert_deep'
    matching_files = [file for file in directory_path.glob(f"*{search_string}*") if file.is_file()]
    val_dataloader = default_test_dataloader_provider()
    for file in matching_files:
        model = PairLossSentenceBERTDeep()
        # the_path = the_path or './checkpoints/SIND_best_20260616_132444_815731pair_loss_bert_best_acc.pth'
        load_checkpoint(model, str(file))
        model.to(DEVICE)
        model.eval()
        # print(f'Testing model from checkpoint: {file}')
        scores = valid_by_pair_head_batched(model, dataloader=val_dataloader)
        print(f'Model: {file}, Test Scores: {scores}')
        logger.warning(f'Model: {file}, Test Scores: {scores}')

def train_and_test():
    train_pair_loss_sentence_bert()
    test_trained()


if __name__ == "__main__":
    train_and_test()