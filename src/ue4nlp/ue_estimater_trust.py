from models.functions import extract_clsvec_predlabels, extract_clsvec_truelabels
from ue4nlp.functions import sep_features_by_class, diffclass_euclid_dist, sameclass_euclid_dist

import numpy as np

class UeEstimatorTrustscore:
    def __init__(self, model, train_dataloader, upper_score, reg_or_class):
        self.model = model
        self.train_dataloader = train_dataloader
        self.upper_score = upper_score
        self.reg_or_class = reg_or_class
        

    def __call__(self, dataloader=None, X_features=None, scores=None):
        if X_features is not None and scores is not None:
            if scores.dtype != np.int32:
                int_scores = np.round(scores * self.upper_score).astype('int32')
            return self._predict_with_fitted_clsvec(X_features, int_scores)
        else:
            X_features, scores = self._extract_features_and_predlabels(dataloader)
            if self.reg_or_class == 'reg':
                int_scores = np.round(scores * self.upper_score).astype('int32')
            else:
                int_scores = scores.astype('int32')
            return self._predict_with_fitted_clsvec(X_features, int_scores)
        
    
    def fit_ue(self):
        if self.reg_or_class == 'reg':
            X_features, y = self._extract_features_and_truelabels(self.train_dataloader)
            int_labels = np.round(y * self.upper_score).astype('int32')
            self.class_features = self._fit_classfeatures(X_features, int_labels)
        else:
            X_features, y = self._extract_features_and_truelabels(self.train_dataloader)
            self.class_features = self._fit_classfeatures(X_features, y.astype('int32'))

        
    def _fit_classfeatures(self, X_features, scores):
        return sep_features_by_class(X_features, scores)
    
    
    def _extract_features_and_predlabels(self, data_loader):
        model = self.model
        X_features, predlabels = extract_clsvec_predlabels(model, data_loader)
        return X_features, predlabels

    
    def _extract_features_and_truelabels(self, data_loader):
        model = self.model
        X_features, truelabels = extract_clsvec_truelabels(model, data_loader)
        return X_features, truelabels

        
    def _predict_with_fitted_clsvec(self, X_features, labels):
        trust_score_values = []
        for x_feature, label in zip(X_features, labels):
            diffclass_dist = diffclass_euclid_dist(x_feature, label, self.class_features)
            sameclass_dist= sameclass_euclid_dist(x_feature, label, self.class_features)
            if sameclass_dist is None:
                trust_score_values = np.append(trust_score_values, 0.)
            else:
                trust_score = diffclass_dist / (diffclass_dist + sameclass_dist)
                trust_score_values = np.append(trust_score_values, trust_score)
        eval_results = {'trust_score': trust_score_values}
        return eval_results