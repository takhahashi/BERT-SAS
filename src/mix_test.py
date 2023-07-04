import os
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
from utils.utils_data import TrainDataModule
from utils.cfunctions import simple_collate_fn, EarlyStopping
from utils.utils_models import create_module
from models.functions import return_predresults
from utils.cfunctions import regvarloss
from models.models import Scaler, Bert, Reg_class_mixmodel
from ue4nlp.ue_estimater_trust import UeEstimatorTrustscore

from utils.dataset import get_upper_score, get_dataset
import json


@hydra.main(config_path="/content/drive/MyDrive/GoogleColab/SA/ShortAnswer/BERT-SAS/configs", config_name="eval_mix")
def main(cfg: DictConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)
    upper_score = get_upper_score(cfg.sas.question_id, cfg.sas.score_id)

    with open(cfg.path.traindata_file_name) as f:
        train_dataf = json.load(f)
    train_dataset = get_dataset(train_dataf, cfg.sas.score_id, upper_score, cfg.model.reg_or_class, tokenizer)
    with open(cfg.path.valdata_file_name) as f:
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

    bert = Bert(cfg.model.model_name_or_path)
    model = Reg_class_mixmodel(bert, num_classes=upper_score+1)
    model.load_state_dict(torch.load(cfg.path.model_save_path))
    eval_results = return_predresults(model, test_dataloader, rt_clsvec=False, dropout=False)

    softmax = nn.Softmax(dim=1)
    pred_int_score = torch.tensor(np.round(eval_results['score'] * upper_score), dtype=torch.int32)
    pred_probs = softmax(torch.tensor(eval_results['logits']))
    mix_trust = pred_probs[torch.arange(len(pred_probs)), pred_int_score]
    eval_results.update({'mix_conf': mix_trust.numpy().copy()})

    max_prob = pred_probs[torch.arange(len(pred_probs)), torch.argmax(pred_probs, dim=-1)]
    eval_results.update({'MP': max_prob.numpy().copy()})

    trust_estimater = UeEstimatorTrustscore(model, 
                                            train_dataloader, 
                                            upper_score,
                                            cfg.model.reg_or_class
                                            )
    trust_estimater.fit_ue()
    trust_results = trust_estimater(test_dataloader)
    eval_results.update(trust_results)


    list_results = {k: v.tolist() for k, v in eval_results.items() if type(v) == type(np.array([1, 2, 3.]))}
    
    with open(cfg.path.results_save_path, mode="wt", encoding="utf-8") as f:
        json.dump(list_results, f, ensure_ascii=False)
    

if __name__ == "__main__":
    main()