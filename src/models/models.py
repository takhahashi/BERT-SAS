import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import gpytorch
from transformers import AutoModel
from utils.cfunctions import regvarloss
from models.spectral_normalization import spectral_norm


class Bert(nn.Module):
    def __init__(self, model_name_or_path):
        super(Bert, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name_or_path)

    def forward(self, dataset):
        outputs = self.bert(dataset['input_ids'], token_type_ids=dataset['token_type_ids'], attention_mask=dataset['attention_mask'])
        sequence_output = outputs['last_hidden_state'][:, 0, :]
        return {'hidden_state': sequence_output}


class BertReg(pl.LightningModule):
    def __init__(
        self,
        bert,
        learning_rate: float,
        criterion=regvarloss,
    ):
        super().__init__()
        self.bert = bert
        self.linear1 = nn.Linear(768, 1)
        self.linear2 = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()
        self.lr = learning_rate
        self.criterion = criterion

    def forward(self, inputs):
        hidden_state = self.bert(inputs)['hidden_state']
        score = self.sigmoid(self.linear1(hidden_state))
        logvar = self.linear2(hidden_state)
        return {'score': score, 'logvar': logvar}

    def training_step(self, batch, batch_idx):
        x = {'input_ids':batch['input_ids'],
             'attention_mask':batch['attention_mask'],
             'token_type_ids':batch['token_type_ids']}
        y = batch['labels']
        outputs = {k: v for k, v in self.forward(x).items()}
        y_hat = outputs['score']
        logvar = outputs['logvar']
        loss = self.criterion(y, y_hat, logvar)
        self.log("train_loss", loss)
        return {"loss": loss, "score": y_hat, "logvar": logvar, "labels": y}

    def validation_step(self, batch, batch_idx):
        x = {'input_ids':batch['input_ids'],
             'attention_mask':batch['attention_mask'],
             'token_type_ids':batch['token_type_ids']}
        y = batch['labels']
        outputs = {k: v for k, v in self.forward(x).items()}
        y_hat = outputs['score']
        logvar = outputs['logvar']
        loss = self.criterion(y, y_hat, logvar)
        self.log("val_loss", loss)
        return {"loss": loss, "score": y_hat, "logvar": logvar, "labels": y}
    
    def test_step(self, batch, batch_idx):
        x = {'input_ids':batch['input_ids'],
             'attention_mask':batch['attention_mask'],
             'token_type_ids':batch['token_type_ids']}
        y = batch['labels']
        outputs = {k: v for k, v in self.forward(x).items()}
        y_hat = outputs['score']
        logvar = outputs['logvar']
        loss = self.criterion(y, y_hat, logvar)
        return {"loss": loss, "score": y_hat, "logvar": logvar, "labels": y}

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr)
    
class BertClass(pl.LightningModule):
    def __init__(
        self,
        bert,
        num_classes,
        learning_rate: float,
        criterion=nn.CrossEntropyLoss(),
    ):
        super().__init__()
        self.bert = bert
        self.linear1 = nn.Linear(768, num_classes)
        self.lr = learning_rate
        self.criterion = criterion

    def forward(self, inputs):
        hidden_state = self.bert(inputs)['hidden_state']
        return {'hidden_state':hidden_state, 'logits': self.linear1(hidden_state)}

    def training_step(self, batch, batch_idx):
        x = {'input_ids':batch['input_ids'],
             'attention_mask':batch['attention_mask'],
             'token_type_ids':batch['token_type_ids']}
        y = batch['labels']
        model_outputs = self.forward(x)
        loss = self.criterion(model_outputs['logits'], y)
        self.log("train_loss", loss)
        y_hat = torch.argmax(model_outputs['logits'], dim=-1)
        return {"loss": loss, "preds": y_hat, "labels": y, "hidden_state": model_outputs['hidden_state'], "logits": model_outputs['logits']}

    def validation_step(self, batch, batch_idx):
        x = {'input_ids':batch['input_ids'],
             'attention_mask':batch['attention_mask'],
             'token_type_ids':batch['token_type_ids']}
        y = batch['labels']
        model_outputs = self.forward(x)
        loss = self.criterion(model_outputs['logits'], y)
        self.log("val_loss", loss)
        y_hat = torch.argmax(model_outputs['logits'], dim=-1)
        return {"loss": loss, "preds": y_hat, "labels": y, "hidden_state": model_outputs['hidden_state'], "logits": model_outputs['logits']}

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr)

class Scaler(torch.nn.Module):
    def __init__(self, init_S=1.0):
        super().__init__()
        self.S = torch.nn.Parameter(torch.tensor([init_S]))

    def forward(self, x):
        return self.S.mul(x)
    
class Reg_class_mixmodel(nn.Module):
    def __init__(self, bert, num_classes):
        super(Reg_class_mixmodel, self).__init__()
        self.bert = bert
        self.linear1 = nn.Linear(768, 1)
        self.linear2 = nn.Linear(768, num_classes)

        self.sigmoid = nn.Sigmoid()

        nn.init.normal_(self.linear1.weight, std=0.02)  # 重みの初期化
        nn.init.normal_(self.linear1.bias, 0)  # バイアスの初期化
        nn.init.normal_(self.linear2.weight, std=0.02)
        nn.init.normal_(self.linear2.bias, 0)  # バイアスの初期化

    def forward(self, dataset):
        hidden_state = self.bert(dataset)['hidden_state']
        score = self.sigmoid(self.linear1(hidden_state))
        logits = self.linear2(hidden_state)
        return {'score': score, 'logits': logits}

    def get_word_vec(self, dataset):        
        outputs = self.bert(dataset['input_ids'], token_type_ids=dataset['token_type_ids'], attention_mask=dataset['attention_mask'])
        sequence_output = outputs['last_hidden_state'][:, 0, :] 
        return {'word_vec': sequence_output}
    
class EscoreScaler(torch.nn.Module):
    def __init__(self, init_S=1.0):
        super().__init__()
        self.S = torch.nn.Parameter(torch.tensor([init_S]))
        self.sigmoid = nn.Sigmoid()
    #e_scaler * class_pred + (1 - e_scaler) * reg_pred
    def right(self, x):
        return (torch.tensor(1.) - self.sigmoid(self.S))*x
    def left(self, x):
        return self.sigmoid(self.S)*x
    
class GPModel(gpytorch.models.ExactGP):
  def __init__(self, train_x, train_y, likelihood, lengthscale=None):
    super(GPModel, self).__init__(train_x, train_y, likelihood)
    self.mean_module = gpytorch.means.ConstantMean()
    self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale_prior=lengthscale))
  def forward(self, x):
    mean_x = self.mean_module(x)
    covar_x = self.covar_module(x)
    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
  
class BertClassSpectralNorm(pl.LightningModule):
    def __init__(
        self,
        bert,
        num_classes,
        learning_rate: float,
        criterion=nn.CrossEntropyLoss(),
    ):
        super().__init__()
        self.bert = bert
        self.linear1 = spectral_norm(nn.Linear(768, num_classes),
                                     n_power_iterations=1, 
                                     norm_bound=0.95)
        self.lr = learning_rate
        self.criterion = criterion

    def forward(self, inputs):
        hidden_state = self.bert(inputs)['hidden_state']
        return {'hidden_state':hidden_state, 'logits': self.linear1(hidden_state)}

    def training_step(self, batch, batch_idx):
        x = {'input_ids':batch['input_ids'],
             'attention_mask':batch['attention_mask'],
             'token_type_ids':batch['token_type_ids']}
        y = batch['labels']
        model_outputs = self.forward(x)
        loss = self.criterion(model_outputs['logits'], y)
        self.log("train_loss", loss)
        y_hat = torch.argmax(model_outputs['logits'], dim=-1)
        return {"loss": loss, "preds": y_hat, "labels": y, "hidden_state": model_outputs['hidden_state'], "logits": model_outputs['logits']}

    def validation_step(self, batch, batch_idx):
        x = {'input_ids':batch['input_ids'],
             'attention_mask':batch['attention_mask'],
             'token_type_ids':batch['token_type_ids']}
        y = batch['labels']
        model_outputs = self.forward(x)
        loss = self.criterion(model_outputs['logits'], y)
        self.log("val_loss", loss)
        y_hat = torch.argmax(model_outputs['logits'], dim=-1)
        return {"loss": loss, "preds": y_hat, "labels": y, "hidden_state": model_outputs['hidden_state'], "logits": model_outputs['logits']}

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr)