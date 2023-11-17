@hydra.main(config_path="/content/drive/MyDrive/GoogleColab/SA/ShortAnswer/BERT-SAS/configs", config_name="eval_acc_config")
def main(cfg: DictConfig):
    model_type 
    five_fold_model_outputs = load_five_fold_results(cfg.sas.prompt_id, cfg.sas.question_id, cfg.sas.score_id, model_type)
    five_fold_trues, five_fold_preds = extract_true_pred(five_fold_model_outputs)
    five_fold_corrs = calc_corr(five_fold_trues, five_fold_preds)
    five_fold_qwks = calc_qwk(five_fold_trues, five_fold_preds)
    five_fold_rmses = calc_rmse(five_fold_trues, five_fold_preds)
    results_dic = {'qwk': np.mean(qwk_arr), 
                    'corr': np.mean(corr_arr), 
                    'rmse': np.mean(rmse_arr)}

    save_path = save_dir_path + model_type
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)
    
def load_five_fold_results(prompt_id, question_id, score_id, model_type):
    five_fold_results = []
    if model_type == 'reg' or model_type == 'mul_reg':
        for fold in range(5):
            with open('/content/drive/MyDrive/GoogleColab//SA/ShortAnswer/{}/{}_results/Reg_{}/fold{}'.format(prompt_id, question_id, score_id, fold)) as f:
                fold_results = json.load(f)
            five_fold_results.append({k: np.array(v) for k, v in fold_results.items()})
    elif model_type == 'class' or model_type == 'mul_class':
        five_fold_results = []
        for fold in range(5):
            with open('/content/drive/MyDrive/GoogleColab//SA/ShortAnswer/{}/{}_results/Class_{}/fold{}'.format(prompt_id, question_id, score_id, fold)) as f:
                fold_results = json.load(f)
            five_fold_results.append({k: np.array(v) for k, v in fold_results.items()})
    elif model_type == 'mix' or model_type == 'mul_mix':
        five_fold_results = []
        for fold in range(5):
            with open('/content/drive/MyDrive/GoogleColab//SA/ShortAnswer/{}/{}_results/Mix_{}/fold{}'.format(prompt_id, question_id, score_id, fold)) as f:
                fold_results = json.load(f)
            five_fold_results.append({k: np.array(v) for k, v in fold_results.items()})
    elif model_type == 'gp':
        five_fold_results = []
        for fold in range(5):
            with open('/content/drive/MyDrive/GoogleColab//SA/ShortAnswer/{}/{}_results/gp_{}/fold{}'.format(prompt_id, question_id, score_id, fold)) as f:
                fold_results = json.load(f)
            five_fold_results.append({k: np.array(v) for k, v in fold_results.items()})
    else:
        raise ValueError(f'`{model_type}` is not valid')

    return five_fold_results

def extract_true_pred(five_fold_results, model_type):
    





if __name__ == "__main__":
    main()