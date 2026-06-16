# 句子对模型分数确实更高，考虑进一步实验
from pair_model import *
import common

DOUBLE_CHECK = False

class PairLossBertWOMLMLoss(PairLossBertV2):
    def forward(self, input_ids, attention_mask, labels):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)  # 获取BERT的输出，直接使用最后一层的CLS向量进行分
        # decode_loss = outputs.loss
        last_hidden_state = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        # 取出[MASK]位置的向量
        mask_token_bool = (input_ids == default_tokenizer().mask_token_id) # [batch_size, seq_len]
        classification_loss = 0.0
        for batch in range(last_hidden_state.size(0)): # 对于每个batch，取出所有mask对应的位置
            # print(default_tokenizer().decode(input_ids[batch]))
            mask_bool_batch = mask_token_bool[batch] # [seq_len]
            mask_indices = torch.where(mask_bool_batch)[0] # [num_masks]
            labels_batch = labels[batch][mask_bool_batch] # [num_masks]
            # print(labels_batch)
            mask_embs = last_hidden_state[batch][mask_bool_batch] # [num_masks, hidden_size]
            # 随机取出两个mask位置的向量进行拼接，送入线性层进行二分类
            idx1, idx2 = torch.randperm(len(mask_indices))[:2] # �
            # print(idx1, idx2)
            # 根据labels判断idx1和idx2的前后关系，构造二分类标签
            label1 = reverse_indexs_tokenized()[labels_batch[idx1].item()] # 将token_id转换回标签索引
            label2 = reverse_indexs_tokenized()[labels_batch[idx2].item()]
            # print(label1, label2)
            pair_label = 1 if label1 < label2 else 0 # 如果label1在label2前面，标签为1，否则为0
            # print(pair_label)
            pair_emb = self.pair_embedding(mask_embs[idx1], mask_embs[idx2]) # size: [hidden_size * 2]
            score = self.pair_classifier(pair_emb) # size: [1]
            # print(score.item())
            print_only_once(labels_batch, idx1, idx2, label1, label2, pair_label, score.item(), input_ids = input_ids[batch])
            square_loss = (score - pair_label) ** 2
            if DOUBLE_CHECK:
                # 反过来也训练一次
                pair_emb_reverse = self.pair_embedding(mask_embs[idx2], mask_embs[idx1]) # size: [hidden_size * 2]
                score_reverse = self.pair_classifier(pair_emb_reverse) # size: [1]
                square_loss += (score_reverse - (1 - pair_label)) ** 2
            classification_loss += square_loss
        classification_loss = classification_loss / last_hidden_state.size(0) # 平均每个batch的损失
        # loss = decode_loss + classification_loss
        return PairLossBertResult(loss=classification_loss, decode_loss=0.0, pair_loss=classification_loss.item())
    

def train_pair_loss_bert():
    model = PairLossBertWOMLMLoss()
    model.to(DEVICE)
    train(epochs=5, model=model, suffix='_pair_loss_bert_womlm')

def test_trained():
    logger = common.get_logger(__name__)
    from pathlib import Path
    directory_path = Path("./checkpoints")
    search_string = '_pair_loss_bert_womlm'
    matching_files = [file for file in directory_path.glob(f"*{search_string}*") if file.is_file()]
    val_dataloader = default_test_dataloader_provider()
    for file in matching_files:
        model = PairLossBertWOMLMLoss()
        # the_path = the_path or './checkpoints/SIND_best_20260616_132444_815731pair_loss_bert_best_acc.pth'
        load_checkpoint(model, str(file))
        model.to(DEVICE)
        model.eval()
        # print(f'Testing model from checkpoint: {file}')
        scores = valid_batched(model, dataloader=val_dataloader)
        print(f'Model: {file}, Test Scores: {scores}')
        logger.warning(f'Model: {file}, Test Scores: {scores}')

def train_and_test():
    train_pair_loss_bert()
    test_trained()


if __name__ == "__main__":
    train_and_test()