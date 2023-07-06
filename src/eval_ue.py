import os

import json
import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer
from utils.utils_data import TrainDataModule
from utils.cfunctions import simple_collate_fn
from utils.utils_models import create_module
from utils.dataset import get_upper_score
from utils.ue_metric_func import calc_rcc_auc, calc_rpp, calc_roc_auc, calc_risk
from models.functions import return_predresults
from ue4nlp.ue_estimater_ensemble import UeEstimatorEnsemble
from ue4nlp.ue_estimater_trust import UeEstimatorTrustscore
from ue4nlp.ue_estimater_mcd import UeEstimatorDp


@hydra.main(config_path="/content/drive/MyDrive/GoogleColab/SA/ShortAnswer/BERT-SAS/configs", config_name="eval_ue_config")
def main(cfg: DictConfig):
    five_fold_results = []
    for fold in range(5):
        with open('/content/drive/MyDrive/GoogleColab//SA/ShortAnswer/Y15/{}_results/Reg_{}/fold{}'.format(cfg.sas.question_id, cfg.sas.score_id, fold)) as f:
            fold_results = json.load(f)
        five_fold_results.append({k: np.array(v) for k, v in fold_results.items()})

    upper_score = get_upper_score(cfg.sas.question_id, cfg.sas.score_id)
    save_dir_path = cfg.path.save_dir_path

    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##simple var####
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = foldr['score']
        uncertainty = foldr['calib_var']
        risk = calc_risk(pred, true, 'reg', upper_score, binary=True)
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(conf=-uncertainty, risk=risk)
        rpp = calc_rpp(conf=-uncertainty, risk=risk)
        roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, reg_or_class='reg', upper_score=upper_score)
        fresults_rcc = np.append(fresults_rcc, rcc_auc)
        fresults_rcc_y.append(rcc_y)
        fresults_roc = np.append(fresults_roc, roc_auc)
        fresults_rpp = np.append(fresults_rpp, rpp)
    results_dic = {'rcc': np.mean(fresults_rcc), 
                   'rpp': np.mean(fresults_rpp), 
                   'roc': np.mean(fresults_roc), 
                   'rcc_y': fresults_rcc_y}
    save_path = save_dir_path + '/simplevar'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)


    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], [] 
    ##reg_trust_score####
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = foldr['score']
        uncertainty = -foldr['trust_score']
        risk = calc_risk(pred, true, 'reg', upper_score, binary=True)
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(conf=-uncertainty, risk=risk)
        rpp = calc_rpp(conf=-uncertainty, risk=risk)
        roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, reg_or_class='reg', upper_score=upper_score)
        fresults_rcc = np.append(fresults_rcc, rcc_auc)
        fresults_rcc_y.append(rcc_y)
        fresults_roc = np.append(fresults_roc, roc_auc)
        fresults_rpp = np.append(fresults_rpp, rpp)
    results_dic = {'rcc': np.mean(fresults_rcc), 
                   'rpp': np.mean(fresults_rpp), 
                   'roc': np.mean(fresults_roc), 
                   'rcc_y': fresults_rcc_y}
    save_path = save_dir_path + '/reg_trust_score'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)


    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##reg_dp###
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = foldr['mcdp_score']
        uncertainty = foldr['calib_mcdp_var']
        risk = calc_risk(pred, true, 'reg', upper_score, binary=True)
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(conf=-uncertainty, risk=risk)
        rpp = calc_rpp(conf=-uncertainty, risk=risk)
        roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, reg_or_class='reg', upper_score=upper_score)
        fresults_rcc = np.append(fresults_rcc, rcc_auc)
        fresults_rcc_y.append(rcc_y)
        fresults_roc = np.append(fresults_roc, roc_auc)
        fresults_rpp = np.append(fresults_rpp, rpp)
    results_dic = {'rcc': np.mean(fresults_rcc), 
                   'rpp': np.mean(fresults_rpp), 
                   'roc': np.mean(fresults_roc), 
                   'rcc_y': fresults_rcc_y}
    save_path = save_dir_path + '/reg_dp'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)



    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##reg_ense###
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = foldr['ense_score']
        uncertainty = foldr['calib_ense_var']
        risk = calc_risk(pred, true, 'reg', upper_score, binary=True)
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(conf=-uncertainty, risk=risk)
        rpp = calc_rpp(conf=-uncertainty, risk=risk)
        roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, reg_or_class='reg', upper_score=upper_score)
        fresults_rcc = np.append(fresults_rcc, rcc_auc)
        fresults_rcc_y.append(rcc_y)
        fresults_roc = np.append(fresults_roc, roc_auc)
        fresults_rpp = np.append(fresults_rpp, rpp)
    results_dic = {'rcc': np.mean(fresults_rcc), 
                   'rpp': np.mean(fresults_rpp), 
                   'roc': np.mean(fresults_roc), 
                   'rcc_y': fresults_rcc_y}
    save_path = save_dir_path + '/reg_mul'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)

    ##class###
    five_fold_results = []
    for fold in range(5):
        with open('/content/drive/MyDrive/GoogleColab//SA/ShortAnswer/Y15/{}_results/Class_{}/fold{}'.format(cfg.sas.question_id, cfg.sas.score_id, fold)) as f:
            fold_results = json.load(f)
        five_fold_results.append({k: np.array(v) for k, v in fold_results.items()})


    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##MP####
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = np.argmax(foldr['logits'], axis=-1)
        uncertainty = -foldr['MP']
        risk = calc_risk(pred, true, 'class', upper_score, binary=True)
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(conf=-uncertainty, risk=risk)
        rpp = calc_rpp(conf=-uncertainty, risk=risk)
        roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, reg_or_class='class', upper_score=upper_score)
        fresults_rcc = np.append(fresults_rcc, rcc_auc)
        fresults_rcc_y.append(rcc_y)
        fresults_roc = np.append(fresults_roc, roc_auc)
        fresults_rpp = np.append(fresults_rpp, rpp)
    results_dic = {'rcc': np.mean(fresults_rcc), 
                   'rpp': np.mean(fresults_rpp), 
                   'roc': np.mean(fresults_roc), 
                   'rcc_y': fresults_rcc_y}
    save_path = save_dir_path + '/MP'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)


    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##class trust####
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = np.argmax(foldr['logits'], axis=-1)
        uncertainty = -foldr['trust_score']
        risk = calc_risk(pred, true, 'class', upper_score, binary=True)
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(conf=-uncertainty, risk=risk)
        rpp = calc_rpp(conf=-uncertainty, risk=risk)
        roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, reg_or_class='class', upper_score=upper_score)
        fresults_rcc = np.append(fresults_rcc, rcc_auc)
        fresults_rcc_y.append(rcc_y)
        fresults_roc = np.append(fresults_roc, roc_auc)
        fresults_rpp = np.append(fresults_rpp, rpp)
    results_dic = {'rcc': np.mean(fresults_rcc), 
                   'rpp': np.mean(fresults_rpp), 
                   'roc': np.mean(fresults_roc), 
                   'rcc_y': fresults_rcc_y}
    save_path = save_dir_path + '/class_trust_score'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)


    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##class dp####
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = foldr['mcdp_score'] 
        uncertainty = foldr['mcdp_uncertainty']
        risk = calc_risk(pred, true, 'class', upper_score, binary=True)
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(conf=-uncertainty, risk=risk)
        rpp = calc_rpp(conf=-uncertainty, risk=risk)
        roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, reg_or_class='class', upper_score=upper_score)
        fresults_rcc = np.append(fresults_rcc, rcc_auc)
        fresults_rcc_y.append(rcc_y)
        fresults_roc = np.append(fresults_roc, roc_auc)
        fresults_rpp = np.append(fresults_rpp, rpp)
    results_dic = {'rcc': np.mean(fresults_rcc), 
                   'rpp': np.mean(fresults_rpp), 
                   'roc': np.mean(fresults_roc), 
                   'rcc_y': fresults_rcc_y}
    save_path = save_dir_path + '/class_dp'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)


    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##class mul####
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = foldr['ense_score']
        uncertainty = foldr['ense_uncertainty'] 
        risk = calc_risk(pred, true, 'class', upper_score, binary=True)
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(conf=-uncertainty, risk=risk)
        rpp = calc_rpp(conf=-uncertainty, risk=risk)
        try:
            roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, reg_or_class='class', upper_score=upper_score)
            fresults_roc = np.append(fresults_roc, roc_auc)
        except:
            print('ROC cannnot calc one class')
        fresults_rcc = np.append(fresults_rcc, rcc_auc)
        fresults_rcc_y.append(rcc_y)
        fresults_rpp = np.append(fresults_rpp, rpp)
    results_dic = {'rcc': np.mean(fresults_rcc), 
                   'rpp': np.mean(fresults_rpp), 
                   'roc': np.mean(fresults_roc), 
                   'rcc_y': fresults_rcc_y}
    save_path = save_dir_path + '/class_mul'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)
    
    five_fold_results = []
    for fold in range(5):
        with open('/content/drive/MyDrive/GoogleColab//SA/ShortAnswer/Y15/{}_results/Mix_{}/fold{}'.format(cfg.sas.question_id, cfg.sas.score_id, fold)) as f:
            fold_results = json.load(f)
        five_fold_results.append({k: np.array(v) for k, v in fold_results.items()})

    save_dir_path = cfg.path.save_dir_path

    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##Mix conf####
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = foldr['score']
        uncertainty = -foldr['mix_conf']
        risk = calc_risk(pred, true, 'reg', upper_score, binary=True)
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(conf=-uncertainty, risk=risk)
        rpp = calc_rpp(conf=-uncertainty, risk=risk)
        roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, reg_or_class='reg', upper_score=upper_score)
        fresults_rcc = np.append(fresults_rcc, rcc_auc)
        fresults_rcc_y.append(rcc_y)
        fresults_roc = np.append(fresults_roc, roc_auc)
        fresults_rpp = np.append(fresults_rpp, rpp)
    results_dic = {'rcc': np.mean(fresults_rcc), 
                    'rpp': np.mean(fresults_rpp), 
                    'roc': np.mean(fresults_roc), 
                    'rcc_y': fresults_rcc_y}
    save_path = save_dir_path + '/mixconf'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)


    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##Mix MP####
    for foldr in five_fold_results:
        true = foldr['labels']
        pred = foldr['score']
        uncertainty = -foldr['MP']
        risk = calc_risk(pred, true, 'reg', upper_score, binary=True)
        rcc_auc, rcc_x, rcc_y = calc_rcc_auc(conf=-uncertainty, risk=risk)
        rpp = calc_rpp(conf=-uncertainty, risk=risk)
        roc_auc = calc_roc_auc(pred, true, conf=-uncertainty, reg_or_class='reg', upper_score=upper_score)
        fresults_rcc = np.append(fresults_rcc, rcc_auc)
        fresults_rcc_y.append(rcc_y)
        fresults_roc = np.append(fresults_roc, roc_auc)
        fresults_rpp = np.append(fresults_rpp, rpp)
    results_dic = {'rcc': np.mean(fresults_rcc), 
                    'rpp': np.mean(fresults_rpp), 
                    'roc': np.mean(fresults_roc), 
                    'rcc_y': fresults_rcc_y}
    save_path = save_dir_path + '/mixMP'
    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(results_dic, f, ensure_ascii=False)

if __name__ == "__main__":
    main()
    