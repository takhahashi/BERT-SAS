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
from utils.dataset import get_Dataset
from utils.ue_metric_func import calc_rcc_auc, calc_rpp, calc_roc_auc, calc_risk
from models.functions import return_predresults
from ue4nlp.ue_estimater_ensemble import UeEstimatorEnsemble
from ue4nlp.ue_estimater_trust import UeEstimatorTrustscore
from ue4nlp.ue_estimater_mcd import UeEstimatorDp


@hydra.main(config_path="/content/drive/MyDrive/GoogleColab/1.AES/ASAP/test1/configs", config_name="eval_config")
def main(cfg: DictConfig):
    train_dataset = get_Dataset(cfg.model.reg_or_class, 
                               cfg.path.traindata_file_name, 
                               cfg.aes.prompt_id, 
                               AutoTokenizer.from_pretrained(cfg.model.model_name_or_path),
                               )
    test_dataset = get_Dataset(cfg.model.reg_or_class, 
                               cfg.path.testdata_file_name, 
                               cfg.aes.prompt_id, 
                               AutoTokenizer.from_pretrained(cfg.model.model_name_or_path),
                               )
    
    if cfg.eval.collate_fn == True:
        collate_fn = simple_collate_fn
    else:
        collate_fn = None

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=cfg.eval.batch_size,
                                                  shuffle=False,
                                                  collate_fn=collate_fn,
                                                  )
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=cfg.eval.batch_size,
                                                  shuffle=False,
                                                  collate_fn=collate_fn,
                                                  )

    low, high = get_score_range(cfg.aes.prompt_id)
    model = create_module(cfg.model.model_name_or_path, 
                          cfg.model.reg_or_class, 
                          learning_rate=1e-5, 
                          num_labels=high-low+1, 
                          )
    model.load_state_dict(torch.load(cfg.path.model_save_path))

    eval_results = return_predresults(model, test_dataloader, rt_clsvec=False, dropout=False)

    softmax = nn.Softmax(dim=1)
    probs = softmax(torch.tensor(eval_results['logits']))
    max_prob = probs[torch.arange(len(probs)), torch.argmax(probs, dim=-1)]
    eval_results.update({'MP': max_prob.numpy().copy()})


    trust_estimater = UeEstimatorTrustscore(model, 
                                            train_dataloader, 
                                            cfg.aes.prompt_id,
                                            cfg.model.reg_or_class,
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
    