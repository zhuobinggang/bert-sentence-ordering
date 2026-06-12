# experiments on ROCStories dataset
import os
import common
import random
from functools import lru_cache
from sind import sind_data_prepare, bert_inputs_to_dataloader_shuffle, train

@lru_cache(maxsize=1)
def mixed_dataset_get():
    import pandas as pd
    the_path1 = os.path.join(common.dataset_base, f'ROCS/ROCStories_spring2016-ROCStories_spring2016.csv')
    the_path2 = os.path.join(common.dataset_base, f'ROCS/ROCStories_winter2017-ROCStories_winter2017.csv')
    df1 = pd.read_csv(the_path1)
    df2 = pd.read_csv(the_path2)
    df = pd.concat([df1, df2], ignore_index=True)
    paragraphs = []
    for _, row in df.iterrows():
        paragraph = [row[f'sentence{i}'] for i in range(1, 6)]
        paragraphs.append(paragraph)
    return paragraphs

@lru_cache(maxsize=1)
def dataset_get():
    paragraphs = mixed_dataset_get()
    random.seed(42)
    random.shuffle(paragraphs)
    random.seed()
    train_ds = paragraphs[:int(0.8*len(paragraphs))]
    val_ds = paragraphs[int(0.8*len(paragraphs)):int(0.9*len(paragraphs))]
    test_ds = paragraphs[int(0.9*len(paragraphs)):]
    return train_ds, val_ds, test_ds

def train_dataloader_provider():
    print('重新制备训练数据集...')
    return bert_inputs_to_dataloader_shuffle(sind_data_prepare(dataset_get()[0]))

@lru_cache(maxsize=1)
def val_dataloader_provider():
    print('重新制备验证数据集...')
    return bert_inputs_to_dataloader_shuffle(sind_data_prepare(dataset_get()[1]))

@lru_cache(maxsize=1)
def test_dataloader_provider():
    print('重新制备测试数据集...')
    return bert_inputs_to_dataloader_shuffle(sind_data_prepare(dataset_get()[2]))

def train_rocs():
    _ = train(epochs=5, suffix='_rocs', trian_dataloader_provider=train_dataloader_provider, val_dataloader_provider=val_dataloader_provider)

if __name__ == '__main__':
    train_rocs()