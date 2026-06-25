def run():
    from critic_topk import valid_trained_in_folder
    valid_trained_in_folder(sind = True, top_k=7)
    valid_trained_in_folder(sind = False, top_k=7)
    from critic_randomk import valid_trained_in_folder
    valid_trained_in_folder(sind = True, npass=4)
    valid_trained_in_folder(sind = False, npass=4)