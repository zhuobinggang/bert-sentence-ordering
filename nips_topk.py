from critic_topk_new import * # 新版top-k算法可以高效应对15x15矩阵的排列问题
# 已经有训练好的critic模型了，直接加载即可
import nips_data
from tqdm import tqdm
from common import add_one, recover_unsorted_paragraph, resort_paragraph, list_in
from sind import get_mask_token_5index_logits, hungarian_algorithm_best_order, DEVICE, indexs_tokenized, cal_tau_acc_pmr, load_checkpoint
from bert_utils import default_tokenizer, default_bert
import random
import common
import nips_bert_input
import torch
from critic_bert_simple import get_critic_score
from nips_critic import default_critic_model_nips

def valid_bert_topk_with_critic_nips(bert, critic, split = 'val', topk = 5, output_details = False):
    paragraphs = nips_data.get_paragraphs(split)
    toker = default_tokenizer()
    all_predicted_labels = []
    all_true_labels = []
    printed = False
    index_dict = indexs_tokenized()
    for paragraph in tqdm(paragraphs):
        best_critic_score = float('-inf')
        best_predicted_labels = None
        labels = None
        # npass次解码
        paragraph_length = len(paragraph)
        # for _ in range(npass):
        random_labels  = add_one(random.sample(range(paragraph_length), paragraph_length)) # 随机打乱标签
        random_paragraph = recover_unsorted_paragraph(paragraph, random_labels)
        bert_input = nips_bert_input.nips_bert_input(random_paragraph, need_shuffle = False) # 代理shuffle
        input_ids = torch.tensor([bert_input.input_ids], dtype=torch.long).to(DEVICE)
        attention_mask = torch.tensor([bert_input.attention_mask], dtype=torch.long).to(DEVICE) # [1, 512]
        label_ids = torch.tensor([bert_input.labels], dtype=torch.long).to(DEVICE) # [1, 512]
        with torch.no_grad():
            logits = bert(input_ids=input_ids, attention_mask=attention_mask).logits # [1, 512, 30522]
        mask_token_bool = (input_ids[0] == toker.mask_token_id)
        predicted_token_ids = logits[0, mask_token_bool] # [n_mask_tokens, vocab_size]
        label_tokens = [index_dict[i] for i in add_one(list(range(len(random_paragraph))))]
        predicted_token_ids = predicted_token_ids[:, label_tokens] # [n_mask_tokens, n_mask_tokens] 每个mask位置对应5个标签的logits
        prob_matrix_numpy = predicted_token_ids.cpu().numpy()
        top5 = get_top_k_permutations_from_matrix(prob_matrix_numpy, top_k=5)
        for temp_predicted_labels in top5:
            temp_resorted_paragraph = resort_paragraph(random_paragraph, temp_predicted_labels)
            critic_score = get_critic_score(critic, temp_resorted_paragraph)
            if critic_score > best_critic_score:
                best_critic_score = critic_score
                best_predicted_labels = temp_predicted_labels
                labels = random_labels
            if not printed:
                print(f'Original paragraph: {paragraph}')
                print(f"Random shuffled paragraph: {random_paragraph}")
                print(f"True label: {random_labels}")
                print(f"Predicted label: {temp_predicted_labels}")
                print(f"Resorted paragraph: {temp_resorted_paragraph}")
                print(f"Critic score: {critic_score}")
                printed = True
        all_predicted_labels.append(best_predicted_labels)
        all_true_labels.append(labels)
    test_result = cal_tau_acc_pmr(all_predicted_labels, all_true_labels, need_fix = False)
    if not output_details:
        return test_result
    else:
        return {
            'tau': test_result.tau,
            'acc': test_result.acc,
            'pmr': test_result.pmr,
            'all_predicted_labels': all_predicted_labels,
            'all_true_labels': all_true_labels
        }
    
    
def valid_trained_in_folder_nips(npass = 3, split = 'test'):
    search_string = 'nips_repeat_'
    matching_files = common.search_files_in_directory(search_string, directory="./checkpoints")
    critic = default_critic_model_nips() 
    taus = []
    accs = []
    pmrs = []
    for file in matching_files:
        bert = default_bert()
        load_checkpoint(bert, str(file))
        bert.to(DEVICE)
        bert.eval()
        result = valid_bert_topk_with_critic_nips(bert, critic, split=split, npass=npass)
        taus.append(result.tau)
        accs.append(result.acc)
        pmrs.append(result.pmr)
        print(f'Model {file}, Test Result: {result}')
        common.logging.warning(f'Model {file}, Test Result: {result}')
    print('Taus:')
    common.cal_mean_std(taus)
    print('Accs:')
    common.cal_mean_std(accs)
    print('PMRs:')
    common.cal_mean_std(pmrs)