import os
import hydra
import torch
import gpytorch
import numpy as np

from omegaconf import DictConfig
from transformers import AutoTokenizer
from utils.cfunctions import simple_collate_fn
from utils.utils_models import create_module
from utils.dataset import get_upper_score, get_dataset
from models.functions import extract_clsvec_truelabels
from models.models import GPModel
import json

@hydra.main(config_path="/content/drive/MyDrive/GoogleColab/SA/ShortAnswer/BERT-SAS/configs", config_name="GP_eval")
def main(cfg: DictConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.scoring_model.model_name_or_path)
    upper_score = get_upper_score(cfg.sas.question_id, cfg.sas.score_id)
    with open(cfg.path.traindata_file_name) as f:
        train_dataf = json.load(f)
    train_dataset = get_dataset(train_dataf, cfg.sas.score_id, upper_score, cfg.scoring_model.reg_or_class, tokenizer)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=16,
                                                    shuffle=True,
                                                    drop_last=False,
                                                    collate_fn=simple_collate_fn)


    num_labels = upper_score + 1
    classifier = create_module(cfg.scoring_model.model_name_or_path,
                          cfg.scoring_model.reg_or_class,
                          1e-5,
                          num_labels=num_labels,
                          )
    classifier = classifier.cuda()
    classifier.load_state_dict(torch.load(cfg.path.scoring_model_savepath), strict=False)
    classifier.eval()

    word_vec, labels = extract_clsvec_truelabels(classifier, train_dataloader)
    train_x = torch.FloatTensor(word_vec)
    train_y = torch.FloatTensor(labels)


    with open(cfg.path.testdata_file_name) as f:
        test_dataf = json.load(f)
    test_dataset = get_dataset(test_dataf, cfg.sas.score_id, upper_score, cfg.scoring_model.reg_or_class, tokenizer)

    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=8,
                                                  shuffle=False,
                                                  collate_fn=simple_collate_fn,
                                                  )

    word_vec, labels = extract_clsvec_truelabels(classifier, test_dataloader)
    test_x = torch.FloatTensor(word_vec)
    test_y = torch.FloatTensor(labels)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPModel(train_x, train_y, likelihood)
    model.load_state_dict(torch.load(cfg.path.GPmodel_save_path))
    likelihood.eval()
    model.eval()

    predictions = model(test_x)
    mean = predictions.mean.cpu().detach().numpy()
    std = predictions.stddev.cpu().detach().numpy()

    eval_results = {'labels':labels, 'score':mean, 'std':std}
    list_results = {k: v.tolist() for k, v in eval_results.items() if type(v) == type(np.array([1, 2, 3.]))}
    
    with open(cfg.path.results_save_path, mode="wt", encoding="utf-8") as f:
        json.dump(list_results, f, ensure_ascii=False)