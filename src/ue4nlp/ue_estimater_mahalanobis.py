from ue4nlp.functions import compute_centroids, compute_covariance, mahalanobis_distance
from models.functions import extract_clsvec_predlabels, extract_clsvec_truelabels
import numpy as np

class UeEstimatorMahalanobis:
    def __init__(self, model, train_dataloader, upper_score, reg_or_class):
        self.model = model
        self.train_dataloader = train_dataloader
        self.reg_or_class = reg_or_class
        self.upper_score = upper_score
        
    def __call__(self, dataloader):
        if self.reg_or_class == 'reg' or self.reg_or_class == 'mix':
            X_features, _ = self._extract_features_and_predlabels(dataloader)
        else: 
            X_features, _ = self._extract_features_and_predlabels(dataloader)
        return self._predict_with_fitted_cov(X_features)
    
    def fit_ue(self):
        if self.reg_or_class == 'reg' or self.reg_or_class == 'mix':
            X_features, y_scaled = self._extract_features_and_truelabels(self.train_dataloader)
            y = np.round(y_scaled * self.upper_score)
        else:
            X_features, y = self._extract_features_and_truelabels(self.train_dataloader)
            
        self.class_cond_centroids = self._fit_centroids(X_features, y.astype('int32'))
        self.class_cond_covariance = self._fit_covariance(X_features, y.astype('int32'))
        print('hidden_state:')
        print(X_features)
        print('------------------')
        print('centroids:', self.class_cond_centroids.shape())
        print(self.class_cond_centroids)
        print('------------------')
        print('cov:', self.class_cond_covariance.shape())
        print(self.class_cond_covariance)
        
    def _fit_covariance(self, X, y):
        centroids = self.class_cond_centroids
        return compute_covariance(centroids, X, y)
        
    def _fit_centroids(self, X, y):
        return compute_centroids(X, y)
    
    def _extract_features_and_predlabels(self, data_loader):
        model = self.model
        X_features, pred_score = extract_clsvec_predlabels(model, data_loader)
        return X_features, pred_score
      
    def _extract_features_and_truelabels(self, data_loader):
        model = self.model
        X_features, truelabels = extract_clsvec_truelabels(model, data_loader)
        return X_features, truelabels
    
    def _predict_with_fitted_cov(self, X_features):
        eval_results = {}
        md = mahalanobis_distance(None, 
                                  None, 
                                  X_features, 
                                  self.class_cond_centroids, 
                                  self.class_cond_covariance)
        eval_results["mahalanobis_distance"] = md
        return eval_results