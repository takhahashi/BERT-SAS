from sklearn.metrics import cohen_kappa_score
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
from utils.dataset import get_upper_score, get_dataset
import json

!pip install omegaconf
from omegaconf import OmegaConf
cfg = OmegaConf.load('../configs/reg_eval.yaml')

cfg.sas.prompt_id='Y15'
cfg.sas.question_id='2-3_2_2'
tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)
for sid in ['A_Score','B_Score','C_Score','D_Score','E_Score']:
  cfg.sas.score_id = sid
  upper_score = get_upper_score(cfg.sas.question_id, cfg.sas.score_id)
  for fold in range(5):
    cfg.eval.fold=fold
    qwk_lis = []
    with open(cfg.path.testdata_file_name) as f:
        test_dataf = json.load(f)
    test_dataset = get_dataset(test_dataf, cfg.sas.score_id, upper_score, cfg.model.reg_or_class, tokenizer)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                                batch_size=cfg.eval.batch_size, 
                                                shuffle=False, 
                                                drop_last=False, 
                                                collate_fn=simple_collate_fn)
    for id in range(5):    
      cfg.model.id=id

      model = create_module(cfg.model.model_name_or_path, 
                            cfg.model.reg_or_class, 
                            learning_rate=1e-5, 
                            num_labels=cfg.model.num_labels, 
                            )
      model.load_state_dict(torch.load(cfg.path.model_save_path), strict=False)


      #dev_results = return_predresults(model, dev_dataloader, rt_clsvec=False, dropout=False)
      eval_results = return_predresults(model, test_dataloader, rt_clsvec=False, dropout=False)
      pred = np.round(eval_results['score'] * upper_score)
      true = np.round(eval_results['labels'] * upper_score)
      qwk = cohen_kappa_score(true, pred, labels = list(range(upper_score+1)), weights='quadratic')
      qwk_lis.append(qwk)
    print(sid, 'Fold', fold, 'QWK', qwk_lis)
  print('-'*10)