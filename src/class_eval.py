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
    
    with open(cfg.path.testdata_file_name) as f:
        test_dataf = json.load(f)
    test_dataset = get_dataset(test_dataf, cfg.sas.score_id, upper_score, cfg.model.reg_or_class, tokenizer)


    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
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

    eval_results = return_predresults(model, test_dataloader, rt_clsvec=False, dropout=False)

    softmax = nn.Softmax(dim=1)
    probs = softmax(torch.tensor(eval_results['logits']))
    max_prob = probs[torch.arange(len(probs)), torch.argmax(probs, dim=-1)]
    eval_results.update({'MP': max_prob.numpy().copy()})


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


    ensemble_estimater = UeEstimatorEnsemble(model, 
                                             cfg.ue.ensemble_model_paths,
                                             cfg.model.reg_or_class,
                                             )
    ensemble_results = ensemble_estimater(test_dataloader)
    eval_results.update(ensemble_results)

    list_results = {}
    for k, v in eval_results.items():
        if type(v) == type(np.array([1, 2.])):
            list_results.update({k: v.tolist()})
        else:
            list_results.update({k: v})

    with open(cfg.path.results_save_path, mode="wt", encoding="utf-8") as f:
        json.dump(list_results, f, ensure_ascii=False)
    

if __name__ == "__main__":
    main()