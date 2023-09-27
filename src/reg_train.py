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
from models.models import Scaler

from utils.dataset import get_upper_score, get_dataset
import json

@hydra.main(config_path="/content/drive/MyDrive/GoogleColab/SA/ShortAnswer/BERT-SAS/configs", config_name="train_reg_config")
def main(cfg: DictConfig):
    cwd = hydra.utils.get_original_cwd()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)
    upper_score = get_upper_score(cfg.sas.question_id, cfg.sas.score_id)


    with open(cfg.path.traindata_file_name) as f:
        train_dataf = json.load(f)
    train_dataset = get_dataset(train_dataf, cfg.sas.score_id, upper_score, cfg.model.reg_or_class, tokenizer)
    with open(cfg.path.valdata_file_name) as f:
        dev_dataf = json.load(f)
    dev_dataset = get_dataset(dev_dataf, cfg.sas.score_id, upper_score, cfg.model.reg_or_class, tokenizer)


    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                    batch_size=cfg.training.batch_size, 
                                                    shuffle=True, 
                                                    drop_last=False, 
                                                    collate_fn=simple_collate_fn)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, 
                                                batch_size=cfg.training.batch_size, 
                                                shuffle=False, 
                                                drop_last=False, 
                                                collate_fn=simple_collate_fn)


    model = create_module(
        cfg.model.model_name_or_path,
        cfg.model.reg_or_class,
        cfg.training.learning_rate,
        )
    optimizer = optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)


    model.train()
    model = model.cuda()
    earlystopping = EarlyStopping(patience=cfg.training.patience, verbose=True, path=cfg.path.save_path)

    scaler = torch.cuda.amp.GradScaler()
    sigma_scaler = Scaler(init_S=1.0).cuda()

    num_train_batch = len(train_dataloader)
    num_dev_batch = len(dev_dataloader)
    for epoch in range(cfg.training.n_epochs):
        train_loss_all = 0
        dev_loss_all = 0
        model.train()
        for idx, t_batch in enumerate(train_dataloader):
            batch = {k: v.cuda() for k, v in t_batch.items()}
            with torch.cuda.amp.autocast():
                training_step_outputs = model.training_step(batch, idx)
            scaler.scale(training_step_outputs['loss']).backward()
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad()

            train_loss_all += training_step_outputs['loss'].to('cpu').detach().numpy().copy()


        ###calibrate_step###
        model.eval()
        with torch.no_grad():
            dev_results = return_predresults(model, dev_dataloader, rt_clsvec=False, dropout=False)
        dev_mu = torch.tensor(dev_results['score']).cuda()
        dev_std = torch.tensor(dev_results['logvar']).exp().sqrt().cuda()
        dev_labels = torch.tensor(dev_results['labels']).cuda()

        # find optimal S
        s_opt = torch.optim.LBFGS([sigma_scaler.S], lr=3e-2, max_iter=2000)

        def closure():
            s_opt.zero_grad()
            loss = regvarloss(y_true=dev_labels, y_pre_ave=dev_mu, y_pre_var=sigma_scaler(dev_std).pow(2).log())
            loss.backward()
            return loss
        s_opt.step(closure)

        for idx, d_batch in enumerate(dev_dataloader):
            batch = {k: v.cuda() for k, v in d_batch.items()}
            dev_step_outputs = model.validation_step(batch, idx)
            dev_mu = dev_step_outputs['score']
            dev_std = dev_step_outputs['logvar'].exp().sqrt()
            dev_labels = dev_step_outputs['labels']
            dev_loss_all += regvarloss(y_true=dev_labels, y_pre_ave=dev_mu, y_pre_var=sigma_scaler(dev_std.cuda()).pow(2).log()).to('cpu').detach().numpy().copy()

        print(f'Epoch:{epoch}, train_loss:{train_loss_all/num_train_batch}, dev_loss:{dev_loss_all/num_dev_batch}')
        earlystopping(dev_loss_all, model)
        if earlystopping.early_stop == True:
            break

if __name__ == "__main__":
    main()