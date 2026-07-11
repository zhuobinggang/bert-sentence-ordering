from bert4so import *
from critic_bert_simple import default_critic_model_nips

def test_trained_one_model_n_pass(model, critic, test_paragraphs, n_pass = 3):
    import critic_bert_simple
    import common 
    total_tau = 0.0
    total_acc = 0.0
    total_pmr = 0.0
    total_count = 0
    
    # 2. 遍历测试集进行微观打分与排序还原
    for tgt_paragraph in test_paragraphs:
        if len(tgt_paragraph) <= 1:
            continue

        predicted_labels_best, true_labels_best = None, None
        best_critic_score = float('-inf')
        exist_predicted_paragraphs = [] # 存储每次解码的 predicted_labels，如果已经有了，就不再评分了

        for _ in range(n_pass):
                
            # 测试时同样需要随机打乱，看模型能否完美复原
            shuffled_paragraph, true_labels = create_shuffled_paragraph_with_labels(tgt_paragraph, need_shuffle=True)
            common.print_only_once(f"labels: {true_labels}")

            input_ids, attention_mask = build_bert_input(shuffled_paragraph)
            input_ids_t = torch.tensor(input_ids).unsqueeze(0).to(DEVICE)
            attention_mask_t = torch.tensor(attention_mask).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                scores = model(input_ids_t, attention_mask_t)
                
            # 将输出的分数转换为绝对位置排名
            scores_cpu = scores.cpu()
            predicted_labels = torch.argsort(torch.argsort(scores_cpu, descending=True)).tolist() # 从大到小排序 -> 从小到大排序 -> 得到绝对位置排名
            resorted_paragraph = common.resort_paragraph(shuffled_paragraph, common.add_one(predicted_labels))

            # 使用 critic 模型评估当前预测的连贯性得分
            # critic_score = ddd!get_critic_score(critic, shuffled_paragraph) # TODO: 替换成使用critic模型
            critic_score = critic_bert_simple.get_critic_score(critic, resorted_paragraph)
            if critic_score > best_critic_score:
                best_critic_score = critic_score
                predicted_labels_best = predicted_labels
                true_labels_best = true_labels

        # 3. 调用你的算分函数
        tau = cal_tau(predicted_labels_best, true_labels_best)
        acc = cal_acc(predicted_labels_best, true_labels_best)
        pmr = cal_PMR(predicted_labels_best, true_labels_best)
        
        if np.isnan(tau):
            tau = 0.0
                
        total_tau += tau
        total_acc += acc
        total_pmr += pmr
        total_count += 1
        
    # 4. 计算当前模型的平均指标
    mean_tau = total_tau / total_count if total_count > 0 else 0
    mean_acc = total_acc / total_count if total_count > 0 else 0
    mean_pmr = total_pmr / total_count if total_count > 0 else 0
    
    return TestResult(mean_tau, mean_acc, mean_pmr)


def test_trained_n_pass(model_files, critic, test_paragraphs, npass = 3):
    taus = []
    accs = []
    pmrs = []
    for model_file in model_files:
        print(f"Testing model: {model_file}")
        model = CriticBert()
        load_checkpoint(model, model_file)
        model.to(DEVICE)
        model.eval()
        test_result = test_trained_one_model_n_pass(model, critic, test_paragraphs, n_pass=npass)
        print(f"Model: {model_file}, Tau: {test_result.tau:.4f}, Acc: {test_result.acc:.4f}, PMR: {test_result.pmr:.4f}")
        taus.append(test_result.tau)
        accs.append(test_result.acc)
        pmrs.append(test_result.pmr)
    print('Taus:')
    common.cal_mean_std(taus)
    print('Accs:')
    common.cal_mean_std(accs)
    print('PMRs:')
    common.cal_mean_std(pmrs)

def test_trained_sind_n_pass(npass = 3):
    search_string = f"sind_listmle_rep"
    # 找出文件夹下所有匹配的 pth/ckpt 文件
    matching_files = common.search_files_in_directory(search_string, directory="./checkpoints")
    if not matching_files:
        print(f"❌ 未找到包含 '{search_string}' 的模型权重文件。")
        return
    test_paragraphs = sind_paragraphs('test')
    critic = default_critic_model_sind()
    test_trained_n_pass(matching_files, critic, test_paragraphs, npass=npass)

def test_trained_rocs_n_pass(npass = 3):
    search_string = f"rocs_listmle_rep"
    # 找出文件夹下所有匹配的 pth/ckpt 文件
    matching_files = common.search_files_in_directory(search_string, directory="./checkpoints")
    if not matching_files:
        print(f"❌ 未找到包含 '{search_string}' 的模型权重文件。")
        return
    test_paragraphs = rocs.dataset_get()['test']
    critic = default_critic_model_rocs()
    test_trained_n_pass(matching_files, critic, test_paragraphs, npass=npass)

def test_trained_nips_n_pass(npass = 3):
    from nips_data import get_paragraphs
    search_string = f"bert4so_nips"
    # 找出文件夹下所有匹配的 pth/ckpt 文件
    matching_files = common.search_files_in_directory(search_string, directory="./checkpoints")
    if not matching_files:
        print(f"❌ 未找到包含 '{search_string}' 的模型权重文件。")
        return
    test_paragraphs = get_paragraphs('test')
    critic = default_critic_model_nips()
    test_trained_n_pass(matching_files, critic, test_paragraphs, npass=npass)