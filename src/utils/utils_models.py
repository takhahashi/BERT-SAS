import torch.optim as optim

from models.models import (
    Bert,
    BertReg,
    BertClass,
    BertClassSpectralNorm
)

def create_module(model_name_or_path, reg_or_class, learning_rate, num_labels=None, save_path=None, spectral_norm=False):

    bert = Bert(model_name_or_path)
    if reg_or_class == 'reg':
        if save_path is not None:
            model = BertReg.load_from_checkpoint(save_path, bert = Bert(model_name_or_path), learning_rate = learning_rate)
        else:
            model = BertReg(bert, learning_rate)
    elif reg_or_class == 'class':
        if save_path is not None:
            if spectral_norm == True:
                model = BertClassSpectralNorm.load_from_checkpoint(save_path, bert = Bert(model_name_or_path), num_labels=num_labels, learning_rate = learning_rate)
            else:
                model = BertClass.load_from_checkpoint(save_path, bert = Bert(model_name_or_path), num_labels=num_labels, learning_rate = learning_rate)
        else:
            if spectral_norm == True:
                model = BertClassSpectralNorm(bert, num_labels, learning_rate)
            else:
                model = BertClass(bert, num_labels, learning_rate)
    else:
        raise ValueError("{} is not a valid value for reg_or_class".format(reg_or_class))
    return model    