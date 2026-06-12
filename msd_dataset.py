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


def train_multi_steps_add_mask4(epochs = 5, suffix = ''):
    def trian_dataloader_provider():
        print('重新制备mask策略的训练数据集...')
        # 准备训练数据
        paragraphs = sind_only_texts_get_by_split('train')
        bert_inputs = sind_data_prepare(paragraphs, 5) # 5 masks一次，4 masks一次，打乱后重新训练试试
        bert_inputs += sind_data_prepare(paragraphs, 4) # 4 masks一次
        train_dataloader = bert_inputs_to_dataloader_shuffle(bert_inputs) # 这里已经打乱了数据顺序
        return train_dataloader
    return train(epochs=epochs, suffix=suffix, trian_dataloader_provider=trian_dataloader_provider)

def train_multi_steps_mask_strategy(epochs = 5, suffix = ''):
    def trian_dataloader_provider():
        print('重新制备mask策略的训练数据集...')
        # 准备训练数据
        paragraphs = sind_only_texts_get_by_split('train')
        bert_inputs = sind_data_prepare(paragraphs, 5) # 5 masks一次， 正常训练
        bert_inputs += sind_data_prepare(paragraphs, 4) # 4 masks一次
        bert_inputs += sind_data_prepare(paragraphs, 3) # 3 masks一次
        bert_inputs += sind_data_prepare(paragraphs, 2) # 2 masks一次
        bert_inputs += sind_data_prepare(paragraphs, 1) # 1 masks一次
        train_dataloader = bert_inputs_to_dataloader_shuffle(bert_inputs) # 这里已经打乱了数据顺序
        return train_dataloader
    return train(epochs=epochs, suffix=suffix, trian_dataloader_provider=trian_dataloader_provider)


if __name__ == "__main__":
    # test_sind_data_prepare(n=2)
    # train_multi_steps(epochs=5, suffix='_mask_strategy_add4')
    train_multi_steps_mask_strategy(epochs=5, suffix='_mask_strategy_all')