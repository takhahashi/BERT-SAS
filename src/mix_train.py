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
from utils.cfunctions import simple_collate_fn, EarlyStopping, ScaleDiffBalance
from utils.utils_models import create_module
from models.functions import return_predresults
from utils.cfunctions import regvarloss, mix_loss1
from models.models import Scaler, Bert, Reg_class_mixmodel

from utils.dataset import get_upper_score, get_dataset
import json
import wandb

@hydra.main(config_path="/content/drive/MyDrive/GoogleColab/SA/ShortAnswer/BERT-SAS/configs", config_name="train_mix_config")
def main(cfg: DictConfig):
    wandb.init(project=cfg.wandb.project,
               name=cfg.wandb.project_name,)
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

    bert = Bert(cfg.model.model_name_or_path)
    model = Reg_class_mixmodel(bert, num_classes=upper_score+1)
    model.train()
    model = model.cuda()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    num_train_batch = len(train_dataloader)
    num_dev_batch = len(dev_dataloader)
    earlystopping = EarlyStopping(patience=cfg.training.patience, path = cfg.path.save_path, verbose = True)

    model.train()
    crossentropy = nn.CrossEntropyLoss()
    mseloss = nn.MSELoss()
    #weight_d = ScaleDiffBalance(num_tasks=2, beta=1.)

    trainloss_list, devloss_list = [], []
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(cfg.training.n_epochs):
        lossall, cross_loss, mse_loss = 0, 0, 0
        devlossall = 0
        model.train()
        for data in train_dataloader:
            data = {k: v.cuda() for k, v in data.items()}
            int_score = torch.round(data['labels'] * upper_score).to(torch.int32).type(torch.LongTensor).cuda()
            with torch.cuda.amp.autocast():
                outputs = model(data)
                #crossentropy_el = crossentropy(outputs['logits'], int_score)
                #mseloss_el = mseloss(outputs['score'].squeeze(), data['labels'])
                #loss, s_wei, diff_wei, alpha, pre_loss = weight_d(crossentropy_el, mseloss_el)
                loss, mse_loss, cross_loss = mix_loss1(data['labels'].squeeze(), outputs['score'].squeeze(), outputs['logits'], high=upper_score, low=0, alpha=100.)
            wandb.log({"epoch":epoch+0.001, "all_loss":loss, "mse_loss":mse_loss, "cross_loss":cross_loss})
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad()
            lossall += loss.to('cpu').detach().numpy().copy()
            #cross_loss += crossentropy_el.to('cpu').detach().numpy().copy()
            #mse_loss += mseloss_el.to('cpu').detach().numpy().copy()

        #trainloss_list = np.append(trainloss_list, lossall/num_train_batch)
        # dev QWKの計算
        model.eval()
        for dev_data in dev_dataloader:
            d_data = {k: v.cuda() for k, v in dev_data.items()}
            int_score = torch.round(d_data['labels'] * upper_score).to(torch.int32).type(torch.LongTensor).to('cpu')
            dev_outputs = {k: v.to('cpu').detach() for k, v in model(d_data).items()}
            #crossentropy_el = crossentropy(dev_outputs['logits'], int_score)
            #mseloss_el = mseloss(dev_outputs['score'].squeeze(), d_data['labels'].to('cpu').detach())
            #loss, s_wei, diff_wei, alpha, pre_loss = weight_d(crossentropy_el, mseloss_el)
            loss, mse_loss, cross_loss = mix_loss1(d_data['labels'].to('cpu').detach().squeeze(), dev_outputs['score'].squeeze(), dev_outputs['logits'], high=upper_score, low=0, alpha=100.)
            devlossall += loss.to('cpu').detach().numpy().copy()
        #devloss_list = np.append(devloss_list, devlossall/num_dev_batch)
        #weight_d.update(lossall/num_train_batch, cross_loss/num_train_batch, mse_loss/num_train_batch)
        wandb.log({"dev_loss":loss})
        print(f'Epoch:{epoch}, train_Loss:{lossall/num_train_batch:.4f}, dev_loss:{devlossall/num_dev_batch:.4f}')
        earlystopping(devlossall/num_dev_batch, model)
        if(earlystopping.early_stop == True): break
    wandb.finish()


if __name__ == "__main__":
    main()