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
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer
from utils.cfunctions import simple_collate_fn
from utils.utils_models import create_module
from models.functions import return_predresults
from ue4nlp.ue_estimater_ensemble import UeEstimatorEnsemble
from ue4nlp.ue_estimater_trust import UeEstimatorTrustscore
from ue4nlp.ue_estimater_mcd import UeEstimatorDp
from ue4nlp.ue_estimater_calibvar import UeEstimatorCalibvar

from utils.dataset import get_upper_score, get_dataset
import json


@hydra.main(config_path="/content/drive/MyDrive/GoogleColab/SA/ShortAnswer/BERT-SAS/configs", config_name="eval_class_config")
def main(cfg: DictConfig):

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)
    upper_score = get_upper_score(cfg.sas.question_id, cfg.sas.score_id)


    with open(cfg.path.traindata_file_name) as f:
        train_dataf = json.load(f)
    train_dataset = get_dataset(train_dataf, cfg.sas.score_id, upper_score, cfg.model.reg_or_class, tokenizer)
    with open(cfg.path.valdata_file_name) as f:
        dev_dataf = json.load(f)
    dev_dataset = get_dataset(dev_dataf, cfg.sas.score_id, upper_score, cfg.model.reg_or_class, tokenizer)
    with open(cfg.path.testdata_file_name) as f:
        test_dataf = json.load(f)
    test_dataset = get_dataset(test_dataf, cfg.sas.score_id, upper_score, cfg.model.reg_or_class, tokenizer)


    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                    batch_size=cfg.eval.batch_size, 
                                                    shuffle=False, 
                                                    drop_last=False, 
                                                    collate_fn=simple_collate_fn)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, 
                                                batch_size=cfg.eval.batch_size, 
                                                shuffle=False, 
                                                drop_last=False, 
                                                collate_fn=simple_collate_fn)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                                batch_size=cfg.eval.batch_size, 
                                                shuffle=False, 
                                                drop_last=False, 
                                                collate_fn=simple_collate_fn)

    
    model = create_module(cfg.model.model_name_or_path, 
                          cfg.model.reg_or_class, 
                          learning_rate=1e-5, 
                          num_labels=upper_score+1, 
                          )
    model.load_state_dict(torch.load(cfg.path.model_save_path))

    dev_results = return_predresults(model, dev_dataloader, rt_clsvec=False, dropout=False)
    eval_results = return_predresults(model, test_dataloader, rt_clsvec=False, dropout=False)

    softmax = nn.Softmax(dim=1)
    probs = softmax(torch.tensor(eval_results['logits']))
    max_prob = torch.argmax(probs, dim=-1).numpy()
    eval_results.update({'MP': max_prob})


    trust_estimater = UeEstimatorTrustscore(model, 
                                            train_dataloader, 
                                            upper_score,
                                            )
    trust_estimater.fit_ue()
    trust_results = trust_estimater(test_dataloader)
    eval_results.update(trust_results)



    mcdp_estimater = UeEstimatorDp(model,
                                   cfg.ue.num_dropout,
                                   cfg.model.reg_or_class,
                                   )
    mcdp_results = mcdp_estimater(test_dataloader)
    eval_results.update(mcdp_results)
    ######calib mcdp var ########
    dev_mcdp_results = mcdp_estimater(dev_dataloader)
    calib_mcdp_var_estimater = UeEstimatorCalibvar(dev_labels=torch.tensor(dev_results['labels']),
                                                   dev_score=torch.tensor(dev_mcdp_results['mcdp_score']),
                                                   dev_logvar=torch.tensor(dev_mcdp_results['mcdp_var']).log(),
                                                   )
    calib_mcdp_var_estimater.fit_ue()
    calib_mcdp_var = calib_mcdp_var_estimater(logvar = torch.tensor(mcdp_results['mcdp_var']).log())
    eval_results.update({'calib_mcdp_var': calib_mcdp_var})




    ensemble_estimater = UeEstimatorEnsemble(cfg.ue.ensemble_model_paths,
                                             cfg.model.reg_or_class,
                                             )
    ensemble_results = ensemble_estimater(test_dataloader)
    eval_results.update(ensemble_results)
    #####calib ense var ##########
    dev_ense_results = ensemble_estimater(dev_dataloader)
    calib_ense_var_estimater = UeEstimatorCalibvar(dev_labels=torch.tensor(dev_results['labels']),
                                                    dev_score=torch.tensor(dev_ense_results['ense_score']),
                                                    dev_logvar=torch.tensor(dev_ense_results['ense_var']).log(),
                                                    )
    calib_ense_var_estimater.fit_ue()
    calib_ense_var = calib_ense_var_estimater(logvar = torch.tensor(ensemble_results['ense_var']).log())
    eval_results.update({'calib_ense_var': calib_ense_var})


    list_results = {k: v.tolist() for k, v in eval_results.items() if type(v) == type(np.array([1, 2, 3.]))}
    
    with open(cfg.path.results_save_path, mode="wt", encoding="utf-8") as f:
        json.dump(list_results, f, ensure_ascii=False)
    

if __name__ == "__main__":
    main()