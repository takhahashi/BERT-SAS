import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt


def down_sample(data, samples=300):
    new_data = [[], []]
    n = len(data[0])
    per_sample = n//samples
    for i in range(n):
        if (i%per_sample == 0) or (i+1 == n):
            new_data[0].append(data[0][i])
            new_data[1].append(data[1][i])
    return new_data


def main():
    ###roc_auc###
    roc_dic = {}
    for utype in ['simplevar', 'reg_dp', 'reg_mul', 'reg_trust_score', 'MP', 'class_dp', 'class_mul', 'class_trust_score']:
        roc_dic[utype] = []
    for qtype in ['1_5']:
        for stype in ['A_Score','B_Score','C_Score','D_Score','E_Score']:
            for utype in ['simplevar', 'reg_dp', 'reg_mul', 'reg_trust_score', 'MP', 'class_dp', 'class_mul', 'class_trust_score']:
                with open('/content/drive/MyDrive/GoogleColab/SA/ShortAnswer/Y15/{}_results/{}/{}'.format(qtype, stype, utype)) as f:
                    fold_results = json.load(f)
                results = {k: np.array(v) for k, v in fold_results.items()}
                roc_dic[utype] = np.append(roc_dic[utype], np.round(results['roc'], decimals=3))
    for k, v in roc_dic.items():
        n_v = np.append(v, np.round(np.mean(v), decimals=3))
        roc_dic[k] = n_v
    roc_table = pd.DataFrame.from_dict(roc_dic, orient='index', columns=['A_Score','B_Score','C_Score','D_Score','E_Score', 'mean'])
    roc_table.to_csv('/content/drive/MyDrive/GoogleColab/SA/ShortAnswer/Y15/{}_results/roc_table.tsv'.format(qtype), sep='\t', index=True)

    ##rpp##
    rpp_dic = {}
    for utype in ['simplevar', 'reg_dp', 'reg_mul', 'reg_trust_score', 'MP', 'class_dp', 'class_mul', 'class_trust_score']:
        rpp_dic[utype] = []
    for qtype in ['1_5']:
        for stype in ['A_Score','B_Score','C_Score','D_Score','E_Score']:
            for utype in ['simplevar', 'reg_dp', 'reg_mul', 'reg_trust_score', 'MP', 'class_dp', 'class_mul', 'class_trust_score']:
                with open('/content/drive/MyDrive/GoogleColab/SA/ShortAnswer/Y15/{}_results/{}/{}'.format(qtype, stype, utype)) as f:
                    fold_results = json.load(f)
                results = {k: np.array(v) for k, v in fold_results.items()}
                rpp_dic[utype] = np.append(rpp_dic[utype], np.round(results['rpp'], decimals=3))
    for k, v in rpp_dic.items():
        n_v = np.append(v, np.round(np.mean(v), decimals=3))
        rpp_dic[k] = n_v
    rpp_table = pd.DataFrame.from_dict(rpp_dic, orient='index', columns=['A_Score','B_Score','C_Score','D_Score','E_Score', 'mean'])
    rpp_table.to_csv('/content/drive/MyDrive/GoogleColab/SA/ShortAnswer/Y15/{}_results/rpp_table.tsv'.format(qtype), sep='\t', index=True)

    ##rcc###
    rcc_dic = {}
    for utype in ['simplevar', 'reg_dp', 'reg_mul', 'reg_trust_score', 'MP', 'class_dp', 'class_mul', 'class_trust_score']:
        rcc_dic[utype] = []
    for qtype in ['1_5']:
        for stype in ['A_Score','B_Score','C_Score','D_Score','E_Score']:
            for utype in ['simplevar', 'reg_dp', 'reg_mul', 'reg_trust_score', 'MP', 'class_dp', 'class_mul', 'class_trust_score']:
                with open('/content/drive/MyDrive/GoogleColab/SA/ShortAnswer/Y15/{}_results/{}/{}'.format(qtype, stype, utype)) as f:
                    fold_results = json.load(f)
                results = {k: np.array(v) for k, v in fold_results.items()}
                rcc_dic[utype] = np.append(rcc_dic[utype], np.round(results['rcc'], decimals=3))
    for k, v in rcc_dic.items():
        n_v = np.append(v, np.round(np.mean(v), decimals=3))
        rcc_dic[k] = n_v
    rcc_table = pd.DataFrame.from_dict(rcc_dic, orient='index', columns=['A_Score','B_Score','C_Score','D_Score','E_Score', 'mean'])
    rcc_table.to_csv('/content/drive/MyDrive/GoogleColab/SA/ShortAnswer/Y15/{}_results/rcc_table.tsv'.format(qtype), sep='\t', index=True)


    ##rcc_y_fig###
    for prompt_id in range(1, 9):
        plt.figure()
        for utype in ['simplevar', 'reg_dp', 'reg_mul', 'reg_trust_score', 'MP', 'class_dp', 'class_mul', 'class_trust_score']:
            with open('/content/drive/MyDrive/GoogleColab/1.AES/ASAP/torchlightning/pt{}/{}'.format(prompt_id, utype)) as f:
                fold_results = json.load(f)
            results = {k: np.array(v) for k, v in fold_results.items()}
            min_len = 1000000
            for rcc_y in results['rcc_y']:
                if len(rcc_y) < min_len:
                    min_len = len(rcc_y)
            rcc_y_list = []
            for rcc_y in results['rcc_y']:
                rcc_y_list.append(np.array(rcc_y)[:min_len])
            mean_rcc_y = np.mean(rcc_y_list, axis=0)
            
            fraction = 1 / len(mean_rcc_y)
            rcc_x = [fraction]
            for i in range(len(mean_rcc_y)-1):
                rcc_x = np.append(rcc_x, fraction+rcc_x[-1])
            down_data = down_sample([rcc_x, mean_rcc_y], samples=50)
            plt.plot(down_data[0], down_data[1], label=utype)
        plt.legend()
        plt.savefig('/content/drive/MyDrive/GoogleColab/1.AES/ASAP/torchlightning/pt{}.png'.format(prompt_id)) 
        plt.show()


if __name__ == "__main__":
    main()