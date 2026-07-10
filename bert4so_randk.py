from bert4so import *

def test_trained_one_model_n_pass(model, test_paragraphs, n_pass = 3):
    
    total_tau = 0.0
    total_acc = 0.0
    total_pmr = 0.0
    total_count = 0
    
    # 2. 遍历测试集进行微观打分与排序还原
    for tgt_paragraph in test_paragraphs:
        if len(tgt_paragraph) <= 1:
            continue
            
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
        predicted_labels = torch.argsort(torch.argsort(scores_cpu, descending=True)).tolist()
        
        # 3. 调用你的算分函数
        tau = cal_tau(predicted_labels, true_labels)
        acc = cal_acc(predicted_labels, true_labels)
        pmr = cal_PMR(predicted_labels, true_labels)
        
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


def test_trained_one_model_n_pass(model, critic, test_paragraphs, n_pass = 3):
    
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
            predicted_labels = torch.argsort(torch.argsort(scores_cpu, descending=True)).tolist()
            
            # 使用 critic 模型评估当前预测的连贯性得分
            critic_score = ddd!get_critic_score(critic, shuffled_paragraph) # TODO: 替换成使用critic模型
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