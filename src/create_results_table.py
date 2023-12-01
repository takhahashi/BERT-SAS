import hydra
import omegaconf
from omegaconf import DictConfig
import numpy as np
import pandas as pd
import json

from utils.dataset import upper_score_dic

@hydra.main(config_path="/content/drive/MyDrive/GoogleColab/SA/ShortAnswer/BERT-SAS/configs", config_name="results_table")
def main(cfg: DictConfig):
    unames = ['reg_default', 'mul_reg_default', 'class_default', 'mul_class_default', 'mix_default', 'mul_mix_default','gp_default', 'class_trust']#, 'class_mahalanobis_spectralnorm_loss_reg_metric', 'class_trust_spectralnorm_loss_reg_metric']
    #print(unames)
    #unames = ['simplevar', 'reg_mul', 'MP', 'class_mul_MP', 'mix', 'mix_mul', 'class_trust_score', 'gp']
    for uncert_metric in ['rcc', 'rpp', 'roc']:
        table = result_table(unames, uncert_metric, cfg.sas.question_id)
    model_names = ['reg', 'mul_reg', 'class', 'mul_class', 'mix', 'mul_mix', 'gp']#, 'class_spectralnorm_loss_reg_metric']
    #model_names = ['simple_reg_acc', 'ense_reg_acc', 'simple_class_acc', 'ense_class_acc', 'mix_acc', 'ense_mix_acc', 'gp_acc']
    for score_acc_metric in ['qwk', 'corr', 'rmse']:
        table = result_table(model_names, score_acc_metric, cfg.sas.question_id)

def result_table(model_or_uncert_types, metric, question_id):
    results_dic = {}
    for m_or_u_type in model_or_uncert_types:
        results_dic[m_or_u_type] = []
    if type(question_id) == omegaconf.listconfig.ListConfig:
        for qid in question_id:
            for m_or_u_type in model_or_uncert_types:
                if qid in ['2-3_1_5', '2-3_2_2', '2-3_2_4']:
                    prompt_id = 'Y15'
                else:
                    prompt_id = 'Y14' 
                with open('/content/drive/MyDrive/GoogleColab/SA/ShortAnswer/{}/{}_results/score/{}'.format(prompt_id, qid, m_or_u_type)) as f:
                    fold_results = json.load(f)
                results = {k: np.array(v) for k, v in fold_results.items()}
                results_dic[m_or_u_type] = np.append(results_dic[m_or_u_type], np.round(results[metric], decimals=3))
        for k, v in results_dic.items():
            n_v = np.append(v, np.round(np.mean(v), decimals=3))
            results_dic[k] = n_v
        table = pd.DataFrame.from_dict(results_dic, orient='index', columns=list(question_id).append('mean'))
        table.to_csv('/content/drive/MyDrive/GoogleColab/SA/ShortAnswer/results_table/all_results_{}_table.tsv'.format(metric), sep='\t', index=True)
    else:
        for score_id in list(upper_score_dic[question_id].keys())[1:]:
            for m_or_u_type in model_or_uncert_types:
                if question_id in ['2-3_1_5', '2-3_2_2', '2-3_2_4']:
                    prompt_id = 'Y15'
                else:
                    prompt_id = 'Y14' 
                with open('/content/drive/MyDrive/GoogleColab/SA/ShortAnswer/{}/{}_results/{}/{}'.format(prompt_id, question_id, score_id, m_or_u_type)) as f:
                    fold_results = json.load(f)
                results = {k: np.array(v) for k, v in fold_results.items()}
                results_dic[m_or_u_type] = np.append(results_dic[m_or_u_type], np.round(results[metric], decimals=3))
        for k, v in results_dic.items():
            n_v = np.append(v, np.round(np.mean(v), decimals=3))
            results_dic[k] = n_v
        table = pd.DataFrame.from_dict(results_dic, orient='index', columns=list(upper_score_dic[question_id].keys())[1:].append('mean'))
        table.to_csv('/content/drive/MyDrive/GoogleColab/SA/ShortAnswer/results_table/{}_{}_table.tsv'.format(question_id, metric), sep='\t', index=True)

    return table

if __name__ == "__main__":
    main()