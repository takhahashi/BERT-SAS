import os

import json
import hydra
import numpy as np
from omegaconf import DictConfig
from utils.dataset import get_upper_score
from utils.utils_eval_acc_ue import load_five_fold_results, extract_true_pred_uncert, check_spectralnorm_regurarization_and_add_path
from sklearn.metrics import cohen_kappa_score

@hydra.main(config_path="/content/drive/MyDrive/GoogleColab/SA/ShortAnswer/BERT-SAS/configs", config_name="eval_acc_config")
def main(cfg: DictConfig):
    model_type = cfg.model.model_type
    save_dir_path = cfg.path.save_dir_path
    upper_score = get_upper_score(cfg.sas.question_id, cfg.sas.score_id)
    five_fold_model_outputs = load_five_fold_results(cfg.sas.prompt_id, cfg.sas.question_id, cfg.sas.score_id, model_type, cfg.model.spectral_norm, cfg.model.reg_metric, cfg.model.reg_cer)
    five_fold_trues, five_fold_preds, _ = extract_true_pred_uncert(five_fold_model_outputs, model_type, uncert_type='default', upper_score=upper_score)
    five_fold_corrs = calc_corr(five_fold_trues, five_fold_preds)
    five_fold_qwks = calc_qwk(five_fold_trues, five_fold_preds, upper_score)
    five_fold_rmses = calc_rmse(five_fold_trues, five_fold_preds)
    results_dic = {'qwk': np.mean(five_fold_qwks), 
                    'corr': np.mean(five_fold_corrs), 
                    'rmse': np.mean(five_fold_rmses)}
    print(five_fold_qwks)

    save_path = save_dir_path + '/' + model_type
    save_path = check_spectralnorm_regurarization_and_add_path(save_path, cfg.model.spectral_norm, cfg.model.reg_metric, cfg.model.reg_cer)
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)

def calc_corr(true, pred):
    if type(true[0]) == np.ndarray:
        corrs = []
        for t, p in zip(true, pred):
            if type(t[0]) != np.int32:
                raise ValueError(f'`{type(t[0])}` is not valid type')
            corrs = np.append(corrs, np.corrcoef(t, p)[0][1])
        return corrs
    else:
        return np.corrcoef(true, pred)[0][1]

def calc_qwk(true, pred, upper_score):
    if type(true[0]) == np.ndarray:
        qwks = []
        for t, p in zip(true, pred):
            if type(t[0]) != np.int32:
                raise ValueError(f'`{type(t[0])}` is not valid type')
            qwks = np.append(qwks, cohen_kappa_score(t, p, labels = list(range(upper_score+1)), weights='quadratic'))
        return qwks
    else:
        return cohen_kappa_score(true, pred, labels = list(range(upper_score+1)), weights='quadratic')
    
def calc_rmse(true, pred):
    if type(true[0]) == np.ndarray:
        rmses = []
        for t, p in zip(true, pred):
            if type(t[0]) != np.int32:
                raise ValueError(f'`{type(t[0])}` is not valid type')
            rmses = np.append(rmses, np.sqrt((t - p) ** 2).mean())
        return rmses
    else:
        return np.sqrt((true - pred) ** 2).mean()

if __name__ == "__main__":
    main()