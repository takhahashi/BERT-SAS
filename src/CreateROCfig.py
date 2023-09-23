import json
import numpy as np
from utils.dataset import get_upper_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

def main():
    five_fold_results = []
    for fold in range(5):
        with open('/content/drive/MyDrive/GoogleColab//SA/ShortAnswer/Y15/{}_results/Class_{}/fold{}'.format('1_5', 'A_Score', fold)) as f:
            fold_results = json.load(f)
        five_fold_results.append({k: np.array(v) for k, v in fold_results.items()})

    upper_score = get_upper_score('1_5', 'A_Score')

    fresults_rcc, fresults_rpp, fresults_roc, fresults_rcc_y = [], [], [], []
    ##simple var####
    for foldr in five_fold_results:
        plt.figure()
        true = foldr['labels']
        pred = np.argmax(foldr['logits'], axis=-1)
        uncertainty = -foldr['MP']
        risk = (true != pred).astype('int32')
        fpr, tpr, thresholds = roc_curve(risk, uncertainty)
        plt.plot(fpr, tpr, marker='o')
        plt.xlabel('FPR: False positive rate')
        plt.ylabel('TPR: True positive rate')
        plt.grid()
        plt.savefig('/content/drive/MyDrive/GoogleColab/SA/ShortAnswer/Y15/{}_results/roc_fig/{}_{}.png'.format('1_5', 'A_Score', fold))

if __name__ == "__main__":
    main()