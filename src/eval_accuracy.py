import os

import json
import hydra
import numpy as np
from omegaconf import DictConfig
from utils.dataset import get_upper_score
from sklearn.metrics import cohen_kappa_score


@hydra.main(config_path="/content/drive/MyDrive/GoogleColab/SA/ShortAnswer/BERT-SAS/configs", config_name="eval_acc_config")
def main(cfg: DictConfig):
    five_fold_results = []
    for fold in range(5):
        with open('/content/drive/MyDrive/GoogleColab//SA/ShortAnswer/Y15/{}_results/Reg_{}/fold{}'.format(cfg.sas.question_id, cfg.sas.score_id, fold)) as f:
            fold_results = json.load(f)
        five_fold_results.append({k: np.array(v) for k, v in fold_results.items()})

    upper_score = get_upper_score(cfg.sas.question_id, cfg.sas.score_id)
    save_dir_path = cfg.path.save_dir_path

    
    corr_arr, qwk_arr, rmse_arr = [], [], []
    ##simple reg####
    for foldr in five_fold_results:
        true = np.round(foldr['labels'] * upper_score).astype('int32')
        pred = np.round(foldr['score'] * upper_score).astype('int32')

        corr_arr = np.append(corr_arr, np.corrcoef(true, pred)[0][1])
        qwk_arr = np.append(qwk_arr, cohen_kappa_score(true, pred, labels = list(range(upper_score+1)), weights='quadratic'))
        rmse_arr = np.append(rmse_arr, np.sqrt((true - pred) ** 2).mean())
    results_dic = {'qwk': np.mean(qwk_arr), 
                    'corr': np.mean(corr_arr), 
                    'rmse': np.mean(rmse_arr)}

    save_path = save_dir_path + '/simple_reg_acc'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)
    

    """
    corr_arr, qwk_arr, rmse_arr = [], [], []
    ##dp reg####
    for foldr in five_fold_results:
        true = np.round(foldr['labels'] * upper_score).astype('int32')
        pred = np.round(foldr['mcdp_score'] * upper_score).astype('int32')

        corr_arr = np.append(corr_arr, np.corrcoef(true, pred)[0][1])
        qwk_arr = np.append(qwk_arr, cohen_kappa_score(true, pred, labels = list(range(upper_score+1)), weights='quadratic'))
        rmse_arr = np.append(rmse_arr, np.sqrt((true - pred) ** 2).mean())
    results_dic = {'qwk': np.mean(qwk_arr), 
                    'corr': np.mean(corr_arr), 
                    'rmse': np.mean(rmse_arr)}

    save_path = save_dir_path + '/dp_reg_acc'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)
    """



    corr_arr, qwk_arr, rmse_arr = [], [], []
    ##ense reg####
    for foldr in five_fold_results:
        true = np.round(foldr['labels'] * upper_score).astype('int32')
        pred = np.round(foldr['ense_score'] * upper_score).astype('int32')

        corr_arr = np.append(corr_arr, np.corrcoef(true, pred)[0][1])
        qwk_arr = np.append(qwk_arr, cohen_kappa_score(true, pred, labels = list(range(upper_score+1)), weights='quadratic'))
        rmse_arr = np.append(rmse_arr, np.sqrt((true - pred) ** 2).mean())
    results_dic = {'qwk': np.mean(qwk_arr), 
                    'corr': np.mean(corr_arr), 
                    'rmse': np.mean(rmse_arr)}

    save_path = save_dir_path + '/ense_reg_acc'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)

 

    ##class###
    five_fold_results = []
    for fold in range(5):
        with open('/content/drive/MyDrive/GoogleColab//SA/ShortAnswer/Y15/{}_results/Class_{}/fold{}'.format(cfg.sas.question_id, cfg.sas.score_id, fold)) as f:
            fold_results = json.load(f)
        five_fold_results.append({k: np.array(v) for k, v in fold_results.items()})


    corr_arr, qwk_arr, rmse_arr = [], [], []
    ##simple_class####
    for foldr in five_fold_results:
        true = foldr['labels'].astype('int32')
        pred = np.argmax(foldr['logits'], axis=-1).astype('int32')

        corr_arr = np.append(corr_arr, np.corrcoef(true, pred)[0][1])
        qwk_arr = np.append(qwk_arr, cohen_kappa_score(true, pred, labels = list(range(upper_score+1)), weights='quadratic'))
        rmse_arr = np.append(rmse_arr, np.sqrt((true - pred) ** 2).mean())
    results_dic = {'qwk': np.mean(qwk_arr), 
                    'corr': np.mean(corr_arr), 
                    'rmse': np.mean(rmse_arr)}

    save_path = save_dir_path + '/simple_class_acc'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)

    """
    corr_arr, qwk_arr, rmse_arr = [], [], []
    ##dp_class####
    for foldr in five_fold_results:
        true = foldr['labels'].astype('int32')
        pred = foldr['mcdp_score'].astype('int32')

        corr_arr = np.append(corr_arr, np.corrcoef(true, pred)[0][1])
        qwk_arr = np.append(qwk_arr, cohen_kappa_score(true, pred, labels = list(range(upper_score+1)), weights='quadratic'))
        rmse_arr = np.append(rmse_arr, np.sqrt((true - pred) ** 2).mean())
    results_dic = {'qwk': np.mean(qwk_arr), 
                    'corr': np.mean(corr_arr), 
                    'rmse': np.mean(rmse_arr)}

    save_path = save_dir_path + '/dp_class_acc'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)
    """

    corr_arr, qwk_arr, rmse_arr = [], [], []
    ##ense_class####
    for foldr in five_fold_results:
        true = foldr['labels'].astype('int32')
        pred = foldr['ense_score'].astype('int32')

        corr_arr = np.append(corr_arr, np.corrcoef(true, pred)[0][1])
        qwk_arr = np.append(qwk_arr, cohen_kappa_score(true, pred, labels = list(range(upper_score+1)), weights='quadratic'))
        rmse_arr = np.append(rmse_arr, np.sqrt((true - pred) ** 2).mean())
    results_dic = {'qwk': np.mean(qwk_arr), 
                    'corr': np.mean(corr_arr), 
                    'rmse': np.mean(rmse_arr)}

    save_path = save_dir_path + '/ense_class_acc'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)

    ##GP####
    five_fold_results = []
    for fold in range(5):
        with open('/content/drive/MyDrive/GoogleColab//SA/ShortAnswer/Y15/{}_results/GP_{}/fold{}'.format(cfg.sas.question_id, cfg.sas.score_id, fold)) as f:
            fold_results = json.load(f)
        five_fold_results.append({k: np.array(v) for k, v in fold_results.items()})

    corr_arr, qwk_arr, rmse_arr = [], [], []
    for foldr in five_fold_results:
        true = np.round(foldr['labels']).astype('int32')
        pred = foldr['score'].astype('int32')
        corr_arr = np.append(corr_arr, np.corrcoef(true, pred)[0][1])
        qwk_arr = np.append(qwk_arr, cohen_kappa_score(true, pred, labels = list(range(upper_score+1)), weights='quadratic'))
        rmse_arr = np.append(rmse_arr, np.sqrt((true - pred) ** 2).mean())
    results_dic = {'qwk': np.mean(qwk_arr), 
                    'corr': np.mean(corr_arr), 
                    'rmse': np.mean(rmse_arr)}
4
    save_path = save_dir_path + '/gp_acc'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)


    ##Mix####
    five_fold_results = []
    for fold in range(5):
        with open('/content/drive/MyDrive/GoogleColab//SA/ShortAnswer/Y15/{}_results/Mix_{}/fold{}'.format(cfg.sas.question_id, cfg.sas.score_id, fold)) as f:
            fold_results = json.load(f)
        five_fold_results.append({k: np.array(v) for k, v in fold_results.items()})

    corr_arr, qwk_arr, rmse_arr = [], [], []
    for foldr in five_fold_results:
        true = np.round(foldr['labels'] * upper_score).astype('int32')
        pred = np.round(foldr['score'] * upper_score).astype('int32')
        corr_arr = np.append(corr_arr, np.corrcoef(true, pred)[0][1])
        qwk_arr = np.append(qwk_arr, cohen_kappa_score(true, pred, labels = list(range(upper_score+1)), weights='quadratic'))
        rmse_arr = np.append(rmse_arr, np.sqrt((true - pred) ** 2).mean())
    results_dic = {'qwk': np.mean(qwk_arr), 
                    'corr': np.mean(corr_arr), 
                    'rmse': np.mean(rmse_arr)}

    save_path = save_dir_path + '/mix_acc'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)

    ##Mix_dp####
    corr_arr, qwk_arr, rmse_arr = [], [], []
    for foldr in five_fold_results:
        true = np.round(foldr['labels'] * upper_score).astype('int32')
        pred = np.round(foldr['mcdp_score'] * upper_score).astype('int32')
        corr_arr = np.append(corr_arr, np.corrcoef(true, pred)[0][1])
        qwk_arr = np.append(qwk_arr, cohen_kappa_score(true, pred, labels = list(range(upper_score+1)), weights='quadratic'))
        rmse_arr = np.append(rmse_arr, np.sqrt((true - pred) ** 2).mean())
    results_dic = {'qwk': np.mean(qwk_arr), 
                    'corr': np.mean(corr_arr), 
                    'rmse': np.mean(rmse_arr)}

    save_path = save_dir_path + '/dp_mix_acc'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)

    ##Mix_ense####
    corr_arr, qwk_arr, rmse_arr = [], [], []
    for foldr in five_fold_results:
        true = np.round(foldr['labels'] * upper_score).astype('int32')
        pred = np.round(foldr['ense_score'] * upper_score).astype('int32')
        corr_arr = np.append(corr_arr, np.corrcoef(true, pred)[0][1])
        qwk_arr = np.append(qwk_arr, cohen_kappa_score(true, pred, labels = list(range(upper_score+1)), weights='quadratic'))
        rmse_arr = np.append(rmse_arr, np.sqrt((true - pred) ** 2).mean())
    results_dic = {'qwk': np.mean(qwk_arr), 
                    'corr': np.mean(corr_arr), 
                    'rmse': np.mean(rmse_arr)}

    save_path = save_dir_path + '/ense_mix_acc'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)

    ##Mix####
    five_fold_results = []
    for fold in range(5):
        with open('/content/drive/MyDrive/GoogleColab//SA/ShortAnswer/Y15/{}_results/Mix_{}/fold{}_expected_score'.format(cfg.sas.question_id, cfg.sas.score_id, fold)) as f:
            fold_results = json.load(f)
        five_fold_results.append({k: np.array(v) for k, v in fold_results.items()})

    corr_arr, qwk_arr, rmse_arr = [], [], []
    for foldr in five_fold_results:
        true = np.round(foldr['labels'] * upper_score).astype('int32')
        pred = np.round(foldr['score'] * upper_score).astype('int32')
        corr_arr = np.append(corr_arr, np.corrcoef(true, pred)[0][1])
        qwk_arr = np.append(qwk_arr, cohen_kappa_score(true, pred, labels = list(range(upper_score+1)), weights='quadratic'))
        rmse_arr = np.append(rmse_arr, np.sqrt((true - pred) ** 2).mean())
    results_dic = {'qwk': np.mean(qwk_arr), 
                    'corr': np.mean(corr_arr), 
                    'rmse': np.mean(rmse_arr)}

    save_path = save_dir_path + '/mix_expected_score_acc'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)
    """
    ##Mix_dp####
    corr_arr, qwk_arr, rmse_arr = [], [], []
    for foldr in five_fold_results:
        true = np.round(foldr['labels'] * upper_score).astype('int32')
        pred = np.round(foldr['mcdp_score'] * upper_score).astype('int32')
        corr_arr = np.append(corr_arr, np.corrcoef(true, pred)[0][1])
        qwk_arr = np.append(qwk_arr, cohen_kappa_score(true, pred, labels = list(range(upper_score+1)), weights='quadratic'))
        rmse_arr = np.append(rmse_arr, np.sqrt((true - pred) ** 2).mean())
    results_dic = {'qwk': np.mean(qwk_arr), 
                    'corr': np.mean(corr_arr), 
                    'rmse': np.mean(rmse_arr)}

    save_path = save_dir_path + '/dp_mix_acc'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)
    """
    ##Mix_ense####
    corr_arr, qwk_arr, rmse_arr = [], [], []
    for foldr in five_fold_results:
        true = np.round(foldr['labels'] * upper_score).astype('int32')
        pred = np.round(foldr['ense_score'] * upper_score).astype('int32')
        corr_arr = np.append(corr_arr, np.corrcoef(true, pred)[0][1])
        qwk_arr = np.append(qwk_arr, cohen_kappa_score(true, pred, labels = list(range(upper_score+1)), weights='quadratic'))
        rmse_arr = np.append(rmse_arr, np.sqrt((true - pred) ** 2).mean())
    results_dic = {'qwk': np.mean(qwk_arr), 
                    'corr': np.mean(corr_arr), 
                    'rmse': np.mean(rmse_arr)}

    save_path = save_dir_path + '/ense_mix_expected_score_acc'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)
        
    ##Mix####
    five_fold_results = []
    for fold in range(5):
        with open('/content/drive/MyDrive/GoogleColab//SA/ShortAnswer/Y15/{}_results/Mix_{}/fold{}_weighted_exp_score'.format(cfg.sas.question_id, cfg.sas.score_id, fold)) as f:
            fold_results = json.load(f)
        five_fold_results.append({k: np.array(v) for k, v in fold_results.items()})

    corr_arr, qwk_arr, rmse_arr = [], [], []
    for foldr in five_fold_results:
        true = np.round(foldr['labels'] * upper_score).astype('int32')
        pred = np.round(foldr['score'] * upper_score).astype('int32')
        corr_arr = np.append(corr_arr, np.corrcoef(true, pred)[0][1])
        qwk_arr = np.append(qwk_arr, cohen_kappa_score(true, pred, labels = list(range(upper_score+1)), weights='quadratic'))
        rmse_arr = np.append(rmse_arr, np.sqrt((true - pred) ** 2).mean())
    results_dic = {'qwk': np.mean(qwk_arr), 
                    'corr': np.mean(corr_arr), 
                    'rmse': np.mean(rmse_arr)}

    save_path = save_dir_path + '/mix_weighted_exp_score_acc'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)

if __name__ == "__main__":
    main()