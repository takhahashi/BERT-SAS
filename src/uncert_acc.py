import os

import json
import hydra
import numpy as np
from omegaconf import DictConfig
from utils.dataset import get_upper_score
from utils.ue_metric_func import calc_rcc_auc, calc_rpp
from utils.utils_eval_acc_ue import load_five_fold_results, extract_true_pred_uncert, check_spectralnorm_regurarization_and_add_path
from sklearn.metrics import roc_auc_score

@hydra.main(config_path="/content/drive/MyDrive/GoogleColab/SA/ShortAnswer/BERT-SAS/configs", config_name="eval_ue_config")
def main(cfg: DictConfig):
    model_type = cfg.model.model_type
    uncert_type = cfg.ue.uncert_type
    save_dir_path = cfg.path.save_dir_path
    upper_score = get_upper_score(cfg.sas.question_id, cfg.sas.score_id)
    five_fold_model_outputs = load_five_fold_results(cfg.sas.prompt_id, cfg.sas.question_id, cfg.sas.score_id, model_type, cfg.model.spectral_norm, cfg.model.reg_metric, cfg.model.reg_cer)
    five_fold_trues, five_fold_preds, five_fold_uncert = extract_true_pred_uncert(five_fold_model_outputs, model_type, uncert_type, upper_score)

    five_fold_rpp, five_fold_roc, five_fold_rcc, five_fold_rcc_y = [], [], [], []
    for idx, (true, pred, uncert) in enumerate(zip(five_fold_trues, five_fold_preds, five_fold_uncert)):
        if type(true[0]) != np.int32 and type(pred[0]) != np.int32:
            raise ValueError(f'`{type(true[0])}` is not valid type')
        five_fold_rpp = np.append(five_fold_rpp, calc_rpp(true, pred, uncert))
        try:
            five_fold_roc = np.append(five_fold_roc, roc_auc_score(true == pred, -uncert))
        except:
            print(f'{cfg.sas.prompt_id} {cfg.sas.question_id} {cfg.sas.score_id} fold{idx} cannnot calc roc')
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(true, pred, -uncert, cfg.rcc.metric_type, upper_score)
        five_fold_rcc = np.append(five_fold_rcc, rcc_auc)
        five_fold_rcc_y.append(rcc_y)
    results_dic = {'rcc': np.mean(five_fold_rcc),
                   'rpp': np.mean(five_fold_rpp),
                   'roc': np.mean(five_fold_roc),
                   'rcc_y': calc_mean_rcc_y(five_fold_rcc_y)}

    save_path = save_dir_path + '/' + model_type + '_' + uncert_type
    save_path = check_spectralnorm_regurarization_and_add_path(save_path, cfg.model.spectral_norm, cfg.model.reg_metric, cfg.model.reg_cer)
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)
    
def calc_mean_rcc_y(rcc_y_lis):
    min_len = len(rcc_y_lis[0])
    for rcc_y in rcc_y_lis:
        if len(rcc_y) < min_len:
            min_len = len(rcc_y)
    rcc_y_arr = []
    for rcc_y in rcc_y_lis:
        rcc_y_arr.append(np.array(rcc_y)[:min_len])
    mean_rcc_y = np.mean(rcc_y_arr, axis=0)
    return mean_rcc_y.tolist()


if __name__ == "__main__":
    main()