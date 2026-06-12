# 数据集制备—5 masks一次，4 masks一次，3 masks一次…… 在生成数据集的时候同时制备额外四个。
from sind import *

def test_sind_data_prepare(n = 2):
    paragraphs = [['First Hello world!', 'Second How are you?', 'Third This is a test.', 'Fourth Bert is great.', 'Fifth I love machine learning.']]
    # 制备数据
    toker = default_tokenizer()
    bert_inputs = sind_data_prepare(paragraphs, random_mask_count=n)
    bi = bert_inputs[0]
    ids = torch.tensor(bi.input_ids)
    ids = ids[ids != toker.pad_token_id] # 去掉padding部分
    print(toker.decode(ids))
    labels = torch.tensor(bi.labels)
    labels = labels[labels != -100] # 只保留被MASK的句子标签
    print(toker.decode(labels))
    print('请检查输出，确认随机MASK的句子标签是否正确！')


def train(epochs = 5, suffix = ''):
    # 1. 准备训练数据
    paragraphs = sind_only_texts_get_by_split('train')
    bert_inputs = sind_data_prepare(paragraphs, 5) # 5 masks一次，4 masks一次，打乱后重新训练试试
    bert_inputs += sind_data_prepare(paragraphs, 4) # 4 masks一次
    train_dataloader = bert_inputs_to_dataloader_shffle(bert_inputs) # 这里已经打乱了数据顺序
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
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    model_suffix = common.get_time_str() + suffix
    MAX_ACC = 0
    for epoch in range(epochs): # 训练指定数量的epoch
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
            writer.global_step += 1
            optimizer.step()
            optimizer.zero_grad()
        score = valid_bert_batched(model, split='val', split_length=256)
        print(f'Validation result after epoch {epoch}: {score}')
        if score.acc > MAX_ACC:
            MAX_ACC = score.acc
            save_checkpoint(model, base_path='checkpoints', epoch=epoch, valid_score=str(score), suffix=f'{model_suffix}_best_acc')
    return model