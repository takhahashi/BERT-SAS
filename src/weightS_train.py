import os
import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import AutoTokenizer
from utils.cfunctions import simple_collate_fn
from models.models import Scaler, Bert, Reg_class_mixmodel, EscoreScaler
from utils.dataset import get_upper_score, get_dataset
import json
    
@hydra.main(config_path="/content/drive/MyDrive/GoogleColab/SA/ShortAnswer/BERT-SAS/configs", config_name="weightS_train")
def main(cfg: DictConfig):
    para_savepath = cfg.path.save_path
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)
    upper_score = get_upper_score(cfg.sas.question_id, cfg.sas.score_id)


    with open(cfg.path.traindata_file_name) as f:
        train_dataf = json.load(f)
    train_dataset = get_dataset(train_dataf, cfg.sas.score_id, upper_score, cfg.model.reg_or_class, tokenizer)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                    batch_size=cfg.training.batch_size, 
                                                    shuffle=True, 
                                                    drop_last=False, 
                                                    collate_fn=simple_collate_fn)

    bert = Bert(cfg.model.model_name_or_path)
    model = Reg_class_mixmodel(bert, num_classes=upper_score+1)
    model.load_state_dict(torch.load(cfg.path.model_save_path))

    model.eval()
    model = model.cuda()

    e_scaler = EscoreScaler(init_S=0.).cuda()

    mseloss = nn.MSELoss()
    s_opt = torch.optim.LBFGS([e_scaler.S], lr=1, max_iter=200)
    all_class_pred = []
    all_reg_pred = []
    all_true_score = []

    train_loss = 0
    for idx, t_batch in enumerate(train_dataloader):
        batch = {k: v.cuda() for k, v in t_batch.items()}
        s_opt.zero_grad()

        int_score = torch.round(batch['labels'] * upper_score).to(torch.float).cuda()
        outputs = {k: v.to('cpu').detach().numpy().copy() for k, v in model(batch).items()}
        class_pred_org = np.argmax(outputs['logits'],axis=1)
        reg_pred = outputs['score'].flatten()
        reg_pred_org = np.round(reg_pred * upper_score)
        all_class_pred.append(class_pred_org)
        all_reg_pred.append(reg_pred_org)
        all_true_score.append(int_score)

    class_pred = torch.tensor(np.concatenate(all_class_pred)).cuda()
    reg_pred = torch.tensor(np.concatenate(all_reg_pred)).cuda()
    train_labels = torch.concat(all_true_score)


    def closure():
        s_opt.zero_grad()
        pred = e_scaler.left(class_pred) + e_scaler.right(reg_pred)
        loss = mseloss(pred, train_labels)
        loss.backward()
        return loss
    mean_pred = (class_pred + reg_pred) / 2
    noscale_loss = mseloss(mean_pred, train_labels)
    s_opt.step(closure)
    scale_pred = e_scaler.left(class_pred) + e_scaler.right(reg_pred)
    scale_loss = mseloss(scale_pred, train_labels)
    print(f'No_s:{noscale_loss}, Apply_S:{scale_loss}, S_Value:{torch.sigmoid(e_scaler.S)}')
    torch.save(e_scaler.state_dict(), para_savepath)

if __name__ == "__main__":
    main()