from functools import lru_cache
from recordclass import recordclass
from typing import List
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BERT_BASE_UNCASED_MODEL_ID = 'google-bert/bert-base-uncased'
ROBERTA_BASE_UNCASED_MODEL_ID = 'FacebookAI/roberta-base'
SpecialTokens = recordclass('SpecialTokens', 'cls sep pad mask unk')
BertInput = recordclass('BertInput', 'input_ids attention_mask labels token_type_ids')
USING_BERT = True

@lru_cache(maxsize=1)
def default_tokenizer():
    from transformers import BertTokenizer,RobertaTokenizer
    if USING_BERT:
        return BertTokenizer.from_pretrained(BERT_BASE_UNCASED_MODEL_ID)
    else:
        return RobertaTokenizer.from_pretrained(ROBERTA_BASE_UNCASED_MODEL_ID)

def default_bert():
    print('重新载入BERT模型...')
    from transformers import BertForMaskedLM, RobertaForMaskedLM
    model = None
    if USING_BERT:
        model = BertForMaskedLM.from_pretrained(BERT_BASE_UNCASED_MODEL_ID)
    else:
        model = RobertaForMaskedLM.from_pretrained(ROBERTA_BASE_UNCASED_MODEL_ID)
    return model.to(DEVICE)

def test_default_bert():
    toker = default_tokenizer()
    bert = default_bert()
    inputs = toker(f"The capital of France is {toker.mask_token}.", return_tensors="pt")
    mask_token_index = (inputs.input_ids == toker.mask_token_id)[0].nonzero(as_tuple=True)[0]
    with torch.no_grad():
        logits = bert(**inputs).logits
    predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    print(toker.decode(predicted_token_id))

@lru_cache(maxsize=1)
def special_tokens_dict():
    toker = default_tokenizer()
    return SpecialTokens(
        cls = toker.cls_token,
        sep = toker.sep_token,
        pad = toker.pad_token,
        mask = toker.mask_token,
        unk = toker.unk_token
    )

@lru_cache(maxsize=None)
def indexs_tokenized(command_length = 100):
    tokenizer = default_tokenizer()
    # command_index_string = ' '.join([str(item) for item in list(range(command_length))]) # 2026.5.24 BUG
    # results =  tokenizer.encode(command_index_string, add_special_tokens = False)
    results = [tokenizer.convert_tokens_to_ids(str(i)) for i in range(command_length)]
    assert len(results) == command_length, f"command_indexs_tokenized: {len(results)} != {command_length}"
    return results

@lru_cache(maxsize=None)
def reverse_indexs_tokenized(command_length = 100):
    indexs_ids = indexs_tokenized(command_length)
    reverse_dict = {token_id: idx for idx, token_id in enumerate(indexs_ids)}
    return reverse_dict