# experiments on ROCStories dataset
from datasets import load_dataset, concatenate_datasets



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