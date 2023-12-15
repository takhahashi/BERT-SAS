import json
import numpy as np

def load_five_fold_results(prompt_id, question_id, score_id, model_type, spectral_norm=False, reg_metric=False, reg_cer=False):
    five_fold_results = []
    if model_type == 'reg' or model_type == 'mul_reg':
        for fold in range(5):
            file_path = '/content/drive/MyDrive/GoogleColab//SA/ShortAnswer/{}/{}_results/Reg_{}/fold{}'.format(prompt_id, question_id, score_id, fold)
            file_path = check_spectralnorm_regurarization_and_add_path(file_path, spectral_norm, reg_metric, reg_cer)
            with open(file_path) as f:
                fold_results = json.load(f)
            five_fold_results.append({k: np.array(v) for k, v in fold_results.items()})
    elif model_type == 'class' or model_type == 'mul_class':
        five_fold_results = []
        for fold in range(5):
            file_path = '/content/drive/MyDrive/GoogleColab//SA/ShortAnswer/{}/{}_results/Class_{}/fold{}'.format(prompt_id, question_id, score_id, fold)
            file_path = check_spectralnorm_regurarization_and_add_path(file_path, spectral_norm, reg_metric, reg_cer)
            with open(file_path) as f:
                fold_results = json.load(f)
            five_fold_results.append({k: np.array(v) for k, v in fold_results.items()})
    elif model_type == 'mix' or model_type == 'mul_mix':
        five_fold_results = []
        for fold in range(5):
            file_path = '/content/drive/MyDrive/GoogleColab//SA/ShortAnswer/{}/{}_results/Mix_{}/fold{}'.format(prompt_id, question_id, score_id, fold)
            file_path = check_spectralnorm_regurarization_and_add_path(file_path, spectral_norm, reg_metric, reg_cer)
            with open(file_path) as f:
                fold_results = json.load(f)
            five_fold_results.append({k: np.array(v) for k, v in fold_results.items()})
    elif model_type == 'gp':
        five_fold_results = []
        for fold in range(5):
            file_path = '/content/drive/MyDrive/GoogleColab//SA/ShortAnswer/{}/{}_results/GP_{}/fold{}'.format(prompt_id, question_id, score_id, fold)
            file_path = check_spectralnorm_regurarization_and_add_path(file_path, spectral_norm, reg_metric, reg_cer)
            with open(file_path) as f:
                fold_results = json.load(f)
            five_fold_results.append({k: np.array(v) for k, v in fold_results.items()})
    elif model_type == 'ordinal_reg':
        five_fold_results = []
        for fold in range(5):
            file_path = '/content/drive/MyDrive/GoogleColab//SA/ShortAnswer/{}/{}_results/Ord_reg_{}/fold{}'.format(prompt_id, question_id, score_id, fold)
            file_path = check_spectralnorm_regurarization_and_add_path(file_path, spectral_norm, reg_metric, reg_cer)
            with open(file_path) as f:
                fold_results = json.load(f)
            five_fold_results.append({k: np.array(v) for k, v in fold_results.items()})
    else:
        raise ValueError(f'`{model_type}` is not valid')

    return five_fold_results

def check_spectralnorm_regurarization_and_add_path(file_path, spectral_norm, reg_metric, reg_cer):
    if spectral_norm == True:
        file_path = file_path + '_spectralnorm'
    if reg_metric == True:
        file_path = file_path + '_loss_reg_metric'
    elif reg_cer == True:
        file_path = file_path + '_loss_reg_cer'
    return file_path

def extract_true_pred_uncert(five_fold_results, model_type, uncert_type, upper_score):
    trues, preds, uncerts = [], [], []
    for fold_result in five_fold_results:
        if model_type == 'reg':
            pred = np.round(fold_result['score'] * upper_score).astype('int32')
            true = np.round(fold_result['labels'] * upper_score).astype('int32')
            if uncert_type == 'default':
                uncert = fold_result['logvar']
            else:
                raise ValueError(f'`{uncert_type}` is not valid')
        elif model_type == 'mul_reg':
            pred = np.round(fold_result['ense_score'] * upper_score).astype('int32')
            true = np.round(fold_result['labels'] * upper_score).astype('int32')
            if uncert_type == 'default':
                uncert = fold_result['ense_var']
            else:
                raise ValueError(f'`{uncert_type}` is not valid')
        elif model_type == 'class' or model_type == 'ordinal_reg':
            pred = np.argmax(fold_result['logits'], axis=-1).astype('int32')
            true = fold_result['labels'].astype('int32')
            if uncert_type == 'default':
                uncert = -fold_result['MP']
            elif uncert_type == 'trust':
                uncert = -fold_result['trust_score']
            elif uncert_type == 'mahalanobis':
                uncert = fold_result['mahalanobis_distance']
            else:
                raise ValueError(f'`{uncert_type}` is not valid')
        elif model_type == 'mul_class':
            pred = fold_result['ense_score'].astype('int32')
            true = fold_result['labels'].astype('int32')
            if uncert_type == 'default':
                uncert = -fold_result['ense_MP']
            else:
                raise ValueError(f'`{uncert_type}` is not valid')
        elif model_type == 'mix':
            pred = np.round(fold_result['labels'] * upper_score).astype('int32')
            true = np.round(fold_result['score'] * upper_score).astype('int32')
            if uncert_type == 'default':
                uncert = -fold_result['mix_conf']
            else:
                raise ValueError(f'`{uncert_type}` is not valid')
        elif model_type == 'mul_mix':
            true = np.round(fold_result['labels'] * upper_score).astype('int32')
            pred = np.round(fold_result['ense_score'] * upper_score).astype('int32')
            if uncert_type == 'default':
                uncert = -fold_result['ense_MP']
            else:
                raise ValueError(f'`{uncert_type}` is not valid')
        elif model_type == 'gp':
            true = np.round(fold_result['labels']).astype('int32')
            pred = np.round(fold_result['score']).astype('int32')
            if uncert_type == 'default':
                uncert = fold_result['std']
            else:
                raise ValueError(f'`{uncert_type}` is not valid')
        else:
            raise ValueError(f'`{model_type}` is not valid')
        uncerts.append(uncert)
        trues.append(true)
        preds.append(pred)
    return trues, preds, uncerts