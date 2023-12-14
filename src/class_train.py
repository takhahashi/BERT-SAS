import os
import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from transformers import AutoTokenizer
from utils.cfunctions import simple_collate_fn, EarlyStopping, compute_loss_cer, compute_loss_metric
from utils.utils_models import create_module
from utils.dataset import get_upper_score, get_dataset
import json


@hydra.main(config_path="/content/drive/MyDrive/GoogleColab/SA/ShortAnswer/BERT-SAS/configs", config_name="train_class_config")
def main(cfg: DictConfig):
    cwd = hydra.utils.get_original_cwd()
    if cfg.model.spectral_norm == True:
        save_path = cfg.path.save_path + '_spectralnorm'
        print('SpectralNorm is applyed!')
        if cfg.training.regularization_metric == True:
            save_path = save_path + '_loss_reg_metric'
            print('loss regularization of metric is applyed!')
        elif cfg.training.regularization_cer == True:
            save_path = save_path + '_loss_reg_cer'
            print('loss regularization of cer is applyed!')
    else:
        save_path = cfg.path.save_path
        print('SpectralNorm is not applyed!')
        if cfg.training.regularization_metric == True:
            save_path = save_path + '_loss_reg_metric'
            print('loss regularization of metric is applyed!')
        elif cfg.training.regularization_cer == True:
            save_path = save_path + '_loss_reg_cer'
            print('loss regularization of cer is applyed!')

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
    num_labels = upper_score + 1
    model = create_module(
        cfg.model.model_name_or_path,
        cfg.model.reg_or_class,
        cfg.training.learning_rate,
        num_labels=num_labels,
        spectral_norm=cfg.model.spectral_norm,
        )
    optimizer = optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)


    model.train()
    model = model.cuda()

    earlystopping = EarlyStopping(patience=cfg.training.patience, verbose=True, path=save_path)

    scaler = torch.cuda.amp.GradScaler()

    num_train_batch = len(train_dataloader)
    num_dev_batch = len(dev_dataloader)
    for epoch in range(cfg.training.n_epochs):
        train_loss_all = dev_loss_all = 0
        for idx, t_batch in enumerate(train_dataloader):
            batch = {k: v.cuda() for k, v in t_batch.items()}
            if idx == 0 and epoch == 0:
                print(batch['labels'])
            with torch.cuda.amp.autocast():
                training_step_outputs = model.training_step(batch, idx)
                if cfg.training.regularization_metric == True:
                    loss = compute_loss_metric(training_step_outputs['hidden_state'],
                                               training_step_outputs['labels'],
                                               training_step_outputs['loss'],
                                               num_labels)
                elif cfg.training.regularization_cer == True:
                    loss = compute_loss_cer(training_step_outputs['logits'],
                                               training_step_outputs['labels'],
                                               training_step_outputs['loss'],
                                               num_labels)
                else:
                    loss = training_step_outputs['loss']
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad()

            train_loss_all += loss.to('cpu').detach().numpy().copy()

        for idx, d_batch in enumerate(dev_dataloader):
            batch = {k: v.cuda() for k, v in d_batch.items()}
            dev_step_outputs = model.validation_step(batch, idx)
            dev_loss_all += dev_step_outputs['loss'].to('cpu').detach().numpy().copy()

        print(f'Epoch:{epoch}, train_loss:{train_loss_all/num_train_batch}, dev_loss:{dev_loss_all/num_dev_batch}')
        earlystopping(dev_loss_all, model)
        if earlystopping.early_stop == True:
            break

if __name__ == "__main__":
    main()