from models.functions import return_predresults
from ue4nlp.functions import compute_mulscore_mulvar, compute_mulMP, compute_mulEntropy, compute_mulprob_epiuncertain
from utils.utils_models import create_module
from ue4nlp.ue_estimater_calibvar import UeEstimatorCalibvar

import torch
    
class UeEstimatorEnsemble:
    def __init__(self, model, model_paths, reg_or_class):
        self.model = model
        self.model_paths = model_paths
        self.reg_or_class = reg_or_class
        
    def __call__(self, dataloader):
        ense_results = self._predict_with_multimodel(dataloader)
        return ense_results
    
    def _multi_pred(self, dataloader):
        mul_results = {}
        model = self.model
        for model_path in self.model_paths:
            model.load_state_dict(torch.load(model_path))
            pred_result = return_predresults(model, dataloader, rt_clsvec=False, dropout=False)
            del pred_result['labels']
            if len(mul_results) == 0:
                mul_results = {k: [v] for k, v in pred_result.items()}
            else:
                for (k1, v1), (k2, v2) in zip(mul_results.items(), pred_result.items()):
                  v1.append(v2)
        return mul_results


    def _predict_with_multimodel(self, dataloader):
        mul_pred_results = self._multi_pred(dataloader)
        mul_num = len(self.model_paths)
        ense_result = {}
        if self.reg_or_class == 'reg':
            mulscore, mulvar = compute_mulscore_mulvar(mul_pred_results['score'], mul_pred_results['logvar'], mul_num)
            ense_result['ense_score'] = mulscore
            ense_result['ense_var'] = mulvar
        else:
            mulscore, mulMP = compute_mulMP(mul_pred_results['logits'], mul_num)
            _, mul_entropy = compute_mulEntropy(mul_pred_results['logits'], mul_num)
            _, epi_uncertainty = compute_mulprob_epiuncertain(mul_pred_results['logits'], mul_num)
            ense_result['ense_score'] = mulscore
            ense_result['ense_MP'] = mulMP
            ense_result['ense_entropy'] = mul_entropy
            ense_result['ense_epi_uncertainty'] = epi_uncertainty
        return ense_result