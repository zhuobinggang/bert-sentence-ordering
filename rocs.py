# experiments on ROCStories dataset
from datasets import load_dataset, concatenate_datasets

def dataset_get():
    ds = load_dataset("mintujupally/ROCStories")
    train_ds = ds['train']
    test_ds = ds['test']
    mixied_ds = concatenate_datasets([train_ds, test_ds])
    # split the mixed dataset into 80% train, 10% val, 10% test
    train_test = mixied_ds.train_test_split(test_size=0.2, seed=42)
    val_test = train_test['test'].train_test_split(test_size=0.5, seed=42)
    train_ds = train_test['train']
    val_ds = val_test['train']
    test_ds = val_test['test']
    return train_ds, val_ds, test_ds

def story_split_prepare():
    import spacy
    nlp = spacy.load("en_core_web_sm")
    ds = load_dataset("mintujupally/ROCStories")
    paragraphs = []
    for item in ds:
        story = item['text']
        doc = nlp(story)
        paragraph = [sent.text for sent in doc.sents]
        assert len(paragraph) == 5, f'每个故事应该有5句话，但发现了{len(paragraph)}句话：{paragraph}'
        paragraphs.append(paragraph)
    # store as jsonl
    import json
    with open('./temp_datasets/rocstories.jsonl', 'w') as f:
        for paragraph in paragraphs:
            json.dump(paragraph, f)
            f.write('\n')


def test_story_split():
    ds = load_dataset("mintujupally/ROCStories")
    for item in ds:
        story = item['text']


