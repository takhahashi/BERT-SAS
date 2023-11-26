import os

import json
import hydra
import numpy as np
from omegaconf import DictConfig
from utils.dataset import get_upper_score
from utils.ue_metric_func import calc_rcc_auc, calc_rpp
from sklearn.metrics import roc_auc_score

@hydra.main(config_path="/content/drive/MyDrive/GoogleColab/SA/ShortAnswer/BERT-SAS/configs", config_name="eval_ue_config")
def main(cfg: DictConfig):
    model_type = cfg.model.model_type
    uncert_type = cfg.ue.uncert_type
    save_dir_path = cfg.path.save_dir_path
    upper_score = get_upper_score(cfg.sas.question_id, cfg.sas.score_id)
    five_fold_model_outputs = load_five_fold_results(cfg.sas.prompt_id, cfg.sas.question_id, cfg.sas.score_id, model_type)
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

def extract_true_pred_uncert(five_fold_results, model_type, uncert_type, upper_score):
    trues, preds, uncerts = [], [], []
    for fold_result in five_fold_results:
        if model_type == 'reg':
            pred = np.round(fold_result['score'] * upper_score).astype('int32')
            true = np.round(fold_result['labels'] * upper_score).astype('int32')
            if uncert_type == 'default':
                uncert = fold_result['logvar']
            else:
                raise ValueError(f'`{uncert_type}` is not valid')
        elif model_type == 'mul_reg':
            pred = np.round(fold_result['ense_score'] * upper_score).astype('int32')
            true = np.round(fold_result['labels'] * upper_score).astype('int32')
            if uncert_type == 'default':
                uncert = fold_result['ense_var']
            else:
                raise ValueError(f'`{uncert_type}` is not valid')
        elif model_type == 'class':
            pred = np.argmax(fold_result['logits'], axis=-1).astype('int32')
            true = fold_result['labels'].astype('int32')
            if uncert_type == 'default':
                uncert = -fold_result['MP']
            elif uncert_type == 'trust':
                uncert = -fold_result['trust_score']
            else:
                raise ValueError(f'`{uncert_type}` is not valid')
        elif model_type == 'mul_class':
            pred = fold_result['ense_score'].astype('int32')
            true = fold_result['labels'].astype('int32')
            if uncert_type == 'default':
                uncert = -fold_result['ense_MP']
            else:
                raise ValueError(f'`{uncert_type}` is not valid')
        elif model_type == 'mix':
            pred = np.round(fold_result['labels'] * upper_score).astype('int32')
            true = np.round(fold_result['score'] * upper_score).astype('int32')
            if uncert_type == 'default':
                uncert = -fold_result['mix_conf']
            else:
                raise ValueError(f'`{uncert_type}` is not valid')
        elif model_type == 'mul_mix':
            true = np.round(fold_result['labels'] * upper_score).astype('int32')
            pred = np.round(fold_result['ense_score'] * upper_score).astype('int32')
            if uncert_type == 'default':
                uncert = -fold_result['ense_MP']
            else:
                raise ValueError(f'`{uncert_type}` is not valid')
        elif model_type == 'gp':
            true = np.round(fold_result['labels']).astype('int32')
            pred = np.round(fold_result['score']).astype('int32')
            if uncert_type == 'default':
                uncert = -fold_result['std']
            else:
                raise ValueError(f'`{uncert_type}` is not valid')
        else:
            raise ValueError(f'`{model_type}` is not valid')
        uncerts.append(uncert)
        trues.append(true)
        preds.append(pred)
    return trues, preds, uncerts


if __name__ == "__main__":
    main()