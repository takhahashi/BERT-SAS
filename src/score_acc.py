import os

import json
import hydra
import numpy as np
from omegaconf import DictConfig
from utils.dataset import get_upper_score
from sklearn.metrics import cohen_kappa_score

@hydra.main(config_path="/content/drive/MyDrive/GoogleColab/SA/ShortAnswer/BERT-SAS/configs", config_name="eval_acc_config")
def main(cfg: DictConfig):
    model_type = cfg.model.model_type
    save_dir_path = cfg.path.save_dir_path
    upper_score = get_upper_score(cfg.sas.question_id, cfg.sas.score_id)
    five_fold_model_outputs = load_five_fold_results(cfg.sas.prompt_id, cfg.sas.question_id, cfg.sas.score_id, model_type)
    five_fold_trues, five_fold_preds = extract_true_pred(five_fold_model_outputs, model_type, upper_score)
    five_fold_corrs = calc_corr(five_fold_trues, five_fold_preds)
    five_fold_qwks = calc_qwk(five_fold_trues, five_fold_preds, upper_score)
    five_fold_rmses = calc_rmse(five_fold_trues, five_fold_preds)
    results_dic = {'qwk': np.mean(five_fold_qwks), 
                    'corr': np.mean(five_fold_corrs), 
                    'rmse': np.mean(five_fold_rmses)}
    print(five_fold_qwks)

    save_path = save_dir_path + '/' + model_type
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
            with open('/content/drive/MyDrive/GoogleColab//SA/ShortAnswer/{}/{}_results/GP_{}/fold{}'.format(prompt_id, question_id, score_id, fold)) as f:
                fold_results = json.load(f)
            five_fold_results.append({k: np.array(v) for k, v in fold_results.items()})
    else:
        raise ValueError(f'`{model_type}` is not valid')

    return five_fold_results

def extract_true_pred(five_fold_results, model_type, upper_score):
    trues, preds = [], []
    for fold_result in five_fold_results:
        if model_type == 'reg':
            pred = np.round(fold_result['score'] * upper_score).astype('int32')
            true = np.round(fold_result['labels'] * upper_score).astype('int32')
        elif model_type == 'mul_reg':
            pred = np.round(fold_result['ense_score'] * upper_score).astype('int32')
            true = np.round(fold_result['labels'] * upper_score).astype('int32')
        elif model_type == 'class':
            pred = np.argmax(fold_result['logits'], axis=-1).astype('int32')
            true = fold_result['labels'].astype('int32')
        elif model_type == 'mul_class':
            pred = fold_result['ense_score'].astype('int32')
            true = fold_result['labels'].astype('int32')
        elif model_type == 'mix':
            pred = np.round(fold_result['labels'] * upper_score).astype('int32')
            true = np.round(fold_result['score'] * upper_score).astype('int32')
        elif model_type == 'mul_mix':
            true = np.round(fold_result['labels'] * upper_score).astype('int32')
            pred = np.round(fold_result['ense_score'] * upper_score).astype('int32')
        elif model_type == 'gp':
            true = np.round(fold_result['labels']).astype('int32')
            pred = np.round(fold_result['score']).astype('int32')
        else:
            raise ValueError(f'`{model_type}` is not valid')
        trues.append(true)
        preds.append(pred)
    return trues, preds

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

def check_spectralnorm_regurarization_and_add_path(file_path, spectral_norm, reg_metric, reg_cer):
    if spectral_norm == True:
        file_path = file_path + '_spectralnorm'
    if reg_metric == True:
        file_path = file_path + '_loss_reg_metric'
    elif reg_cer == True:
        file_path = file_path + '_loss_reg_cer'
    return file_path

if __name__ == "__main__":
    main()