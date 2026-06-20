from two_pass_plus import *
from critic_bert_simple import *
from critic_bert import resort_paragraph, recover_unsorted_paragraph

def valid_bert_n_pass_random_with_critic(bert, critic, split = 'val', npass = 3, paragraphs = None):
    if paragraphs is None:
        paragraphs = sind_paragraphs(split)
    all_predicted_labels = []
    all_true_labels = []
    printed = False
    for paragraph in tqdm(paragraphs):
        best_critic_score = float('-inf')
        best_predicted_labels = None
        labels = None
        # npass次解码
        for _ in range(npass):
            random_labels  = add_one(random.sample(range(5), 5))
            random_paragraph = recover_unsorted_paragraph(paragraph, random_labels)
            bert_input = create_bert_input_for_shuffled_paragraph(random_paragraph, random_labels)
            temp_mask_token_5index_logits = get_mask_token_5index_logits(bert_input.input_ids, bert_input.attention_mask, bert)
            temp_predicted_labels = hungarian_algorithm_best_order(temp_mask_token_5index_logits.cpu().numpy())
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
    # 在修正标签之前计算一次
    test_result = cal_tau_acc_pmr(all_predicted_labels, all_true_labels, need_fix = False)
    return test_result

def default_trained_bert():
    bert = default_bert()
    load_checkpoint(bert, './checkpoints/SIND_best_e1.pth' )
    bert.to(DEVICE)
    bert.eval()
    return bert

def valid_trained():
    critic = default_critic_model_sind()
    bert = default_trained_bert()
    valid_bert_n_pass_random_with_critic(bert, critic, 'test', npass=3)
    
def valid_trained_in_folder(sind = True):
    search_string = '_vanilla_sind_' if sind else '_vanilla_rocs_'
    critic = default_critic_model_sind() if sind else default_critic_model_rocs()
    paragraphs = sind_paragraphs('test') if sind else rocs.dataset_get()['test']
    taus = []
    accs = []
    pmrs = []
    from pathlib import Path
    directory_path = Path("./checkpoints")
    matching_files = [file for file in directory_path.glob(f"*{search_string}*") if file.is_file()]
    for file in matching_files:
        bert = default_bert()
        load_checkpoint(bert, str(file))
        bert.to(DEVICE)
        bert.eval()
        result = valid_bert_n_pass_random_with_critic(bert, critic, split='do_not_use', npass=3, paragraphs=paragraphs)
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

