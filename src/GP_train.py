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

@hydra.main(config_path="/content/drive/MyDrive/GoogleColab/SA/ShortAnswer/BERT-SAS/configs", config_name="GP_train")
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
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPModel(train_x, train_y, likelihood)
    training_iter = cfg.training.iter_num
    model.train()
    likelihood.train()
    model.covar_module.base_kernel.lengthscale = np.linalg.norm(train_x[0].numpy() - train_x[1].numpy().T) ** 2 / 2

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=cfg.training.lr)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f lengthscale: %.3f noise: %.3f' % (
            i+1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()

    torch.save(model.state_dict(), cfg.path.save_path)

if __name__ == "__main__":
    main()