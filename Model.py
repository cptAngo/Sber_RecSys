import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle as pkl

class Model:
    def __init__(self, parameters=None, random_state=432):
        if parameters != None:
            self._parameters = parameters.copy()
            self._random_state = random_state
            self._lgbmc = lgb.LGBMClassifier(objective='binary', random_state=self._random_state, 
                                 boosting_type=self._parameters['boosting_type'],
                                 colsample_bytree=self._parameters['colsample_bytree'],
                                 learning_rate=self._parameters['learning_rate'],
                                 max_depth=self._parameters['max_depth'],
                                 min_child_samples=self._parameters['min_child_samples'],
                                 min_child_weight=self._parameters['min_child_weight'],
                                 min_split_gain=self._parameters['min_split_gain'],
                                 n_estimators=self._parameters['n_estimators'],
                                 num_leaves=self._parameters['num_leaves'],
                                 reg_alpha=self._parameters['reg_alpha'],
                                 reg_lambda=self._parameters['reg_lambda'],
                                 subsample=self._parameters['subsample'],
                                 subsample_for_bin=self._parameters['subsample_for_bin'],
                                 subsample_freq=self._parameters['subsample_freq'])
        else:
            self._parameters = None
            self._random_state = random_state
            self._lgbmc = lgb.LGBMClassifier(random_state=self._random_state)
            
        self._model_lgbm = None
        
        
    def train(self, X, y):
        self._model_lgbm = self._lgbmc.fit(X, y)
        
    def predict_proba(self, X):
        return self._model_lgbm.predict_proba(X)
    
    def save_model(self, path):
        with open(path, 'wb') as f:
            pkl.dump(self, f)
            
    def load_model(self, path):
        with open(path, 'rb') as f:
            new_model = pkl.load(f)
            self._parameters = new_model._get_params()
            self._random_state = new_model._get_rstate()
            self._lgbmc = new_model._get_model()
            self._model_lgbm = new_model._get_trained_model()
            
    def feature_importance(self):
        if self._model_lgbm != None:
            lgb.plot_importance(self._model_lgbm)
        else:
            print('Model hasn\'t been fitted!')
        
    def _get_params(self):
        return self._parameters
    
    def _get_rstate(self):
        return self._random_state
    
    def _get_model(self):
        return self._lgbmc
    
    def _get_trained_model(self):
        return self._model_lgbm
        