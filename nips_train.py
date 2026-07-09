from sind import *
from nips_data import get_paragraphs
from nips_bert_input import nips_bert_input


# 使用匈牙利算法解码得到标签
def valid_bert_nips(bert, split = 'val'):
    if bert is None:
        bert = default_bert()
    bert.eval()
    toker = default_tokenizer()
    # 首先将val数据集转换成BertInput格式
    paragraphs = get_paragraphs(split)
    # 然后使用默认的BERT模型进行解码，计算准确率和tau值
    all_predicted_labels = []
    all_true_labels = []
    reversed_dict = reverse_indexs_tokenized()
    index_dict = indexs_tokenized()
    # index_1_to_5_token_ids = [index_dict[i] for i in range(1, 6)]
    random.seed(42)
    for paragraph in paragraphs:
        bert_input = nips_bert_input(paragraph, need_shuffle = True)
        input_ids = torch.tensor([bert_input.input_ids], dtype=torch.long).to(DEVICE)
        attention_mask = torch.tensor([bert_input.attention_mask], dtype=torch.long).to(DEVICE) # [1, 512]
        label_ids = torch.tensor([bert_input.labels], dtype=torch.long).to(DEVICE) # [1, 512]
        with torch.no_grad():
            logits = bert(input_ids=input_ids, attention_mask=attention_mask).logits # [1, 512, 30522]
        mask_token_bool = (input_ids[0] == toker.mask_token_id)
        predicted_token_ids = logits[0, mask_token_bool] # [n_mask_tokens, vocab_size]
        label_tokens = [index_dict[i] for i in add_one(list(range(len(paragraph))))]
        predicted_token_ids = predicted_token_ids[:, label_tokens] # [n_mask_tokens, n_mask_tokens] 每个mask位置对应5个标签的logits
        predicted_labels = hungarian_algorithm_best_order(predicted_token_ids.cpu().numpy()) # [n_mask_tokens] 每个位置的最终标签（1-5）
        true_label_ids = label_ids[0][label_ids[0] != -100] # [n_mask_tokens]
        assert len(predicted_token_ids) == len(true_label_ids), "There should be the same number of predicted and true labels"
        true_labels = [reversed_dict[b.item()] for b in true_label_ids]
        all_predicted_labels.append(predicted_labels)
        all_true_labels.append(true_labels)
    # 在修正标签之前计算一次
    test_result = cal_tau_acc_pmr(all_predicted_labels, all_true_labels, need_fix = False)
    # print("Now fixing predicted labels and recalculating metrics...")
    # _ = cal_tau_acc_pmr(all_predicted_labels, all_true_labels, need_fix = True)
    random.seed()
    return test_result

def train_dataloader_provider():
    paragraphs = get_paragraphs('train')
    bert_inputs = []
    for paragraph in paragraphs:
        bert_input = nips_bert_input(paragraph, need_shuffle = True)
        bert_inputs.append(bert_input)
    dataloader = bert_inputs_to_dataloader_shuffle(bert_inputs)
    return dataloader

# NOTE: NIPS默认训练10个epoch，其他数据集默认训练5个epoch
def train(epochs = 10, suffix = '_nips'):
    # 记录日志
    logger = common.logging.getLogger(__name__)
    writer = common.get_writer()
    # Train
    from accelerate import Accelerator
    accelerator = Accelerator()
    # model.cuda()
    model = default_bert()
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    model, optimizer = accelerator.prepare(
        model, optimizer
    )
    model_suffix = common.get_time_str() + suffix
    MAX_TAU = 0
    for epoch in range(epochs): # 训练指定数量的epoch
        train_dataloader = accelerator.prepare(train_dataloader_provider())
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            if batch_idx % 1000 == 0:
                logger.warning(f'{common.get_time_str()} Training iteration {batch_idx}')
            input_ids, attention_mask, label_ids = batch
            # NOTE: 2025.5.11 RoBERTa don't use token_type_ids! Error happens if use it!
            outputs = model(input_ids=input_ids.to(DEVICE), 
                    attention_mask=attention_mask.to(DEVICE),
                    labels=label_ids.to(DEVICE))
            loss = outputs.loss
            accelerator.backward(loss)
            writer.add_scalar(f'Loss', loss.item(), writer.global_step)
            if hasattr(model, 'pair_classifier'):
                writer.add_scalar(f'Pair_Loss', outputs.pair_loss, writer.global_step)
            writer.global_step += 1
            optimizer.step()
            optimizer.zero_grad()
        model.eval()
        score = valid_bert_nips(model, split = 'val')
        model.train()
        if score.tau > MAX_TAU:
            print(f'保存模型，当前tau提升到{score.tau}，之前的最高tau是{MAX_TAU}')
            MAX_TAU = score.tau
            save_checkpoint(model, base_path='checkpoints', epoch=epoch, valid_score=str(score), suffix=f'{model_suffix}_best_tau', prefix="NIPS_best")


def n_repeat_train(epochs = 10, n_repeat = 3):
    for i in range(n_repeat):
        print(f'第{i+1}次训练...')
        train(epochs = epochs, suffix = f'nips_repeat_{i+1}')


def checkpoint_paths():
    from pathlib import Path
    directory_path = Path("./checkpoints")
    search_string = 'nips_repeat_'
    matching_files = [file for file in directory_path.glob(f"*{search_string}*") if file.is_file()]
    return matching_files

def test_trained():
    ckpts = checkpoint_paths()
    for file in ckpts:
        bert = default_bert()
        load_checkpoint(bert, str(file)) # 已默认将模型移动到DEVICE上并设置为eval模式
        result = valid_bert_nips(bert, 'test')
        print(f'Model nips repeat: {file}, Test Result: {result}')
        common.logging.warning(f'Model nips repeat: {file}, Test Result: {result}')