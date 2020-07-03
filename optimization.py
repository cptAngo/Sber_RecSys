import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import lightgbm as lgb
import pickle as pkl
from pandas.api.types import CategoricalDtype
import os
import configparser
from Model import Model
import sys

random_state = 432 # Для повторяемости результатов

feat_cols = ['user_region', 'product_category', 'price', 'asset', 'user_total_buy', 'user_dif_cat', 'product_total_buy', 'prod_dif_region', 'label'] # Колонки для обучения и метка

# Функция для считывания параметров для оптимизации из конфигурационного файла
def read_grid_params_from_config(path):
    config = configparser.ConfigParser()
    config.read(path)
    
    params = {}
    
    for section in config.sections():
        for el in config[section]:
            line = config[section][el]
            if el == 'boosting_type':
                params[el] = line.split(', ')
            elif el in ['max_depth', 'min_child_samples', 'n_estimators', 'num_leaves', 'subsample_for_bin', 'subsample_freq']:
                params[el] = np.fromstring(line, dtype=int, sep=', ')
            else:
                params[el] = np.fromstring(line, dtype=float, sep=', ')
                    
    return params

def average_precision(bought_products, sorted_recommended_products, K):
    av_prec = 0
    recommends = list(map(lambda x: x in bought_products, sorted_recommended_products))
    precision = [sum(recommends[:k+1])/(k+1.) for k in range(K)]
    for i in range(K):
        av_prec += recommends[i]*precision[i]
        
    return av_prec/len(bought_products)

if __name__=='__main__':
    
    dir_path = str(sys.argv[1]) # Путь к папке с обучающим, валидационным, тестовым и различными вспомогательными датасетами, которые получились в результате выполнения скрипта preprocess.py
    config_path = str(sys.argv[2]) # Путь к конфигурационному файлу для подбора гиперпараметров
    
    train_df = pd.read_csv(dir_path + '/train_dataset.csv')
    val_df = pd.read_csv(dir_path + '/val_df.csv')
    all_prods = pd.read_csv(dir_path + '/all_prods.csv')
    all_users = pd.read_csv(dir_path + '/all_users.csv')
    
    train_df['user_region'] = train_df['user_region'].astype(CategoricalDtype(categories=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))
    train_df['product_category'] = train_df['product_category'].astype('category')
    
    val_df['user_region'] = val_df['user_region'].astype(CategoricalDtype(categories=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))
    val_df['bought_products'] = val_df['bought_products'].apply(lambda x:np.fromstring(x.strip('[ ]'), dtype=int, sep=' '))
    
    all_prods['product_category'] = all_prods['product_category'].astype('category')
    
    all_users['user_region'] = all_users['user_region'].astype(CategoricalDtype(categories=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))
    
    # Считываем значения гиперпараметров для оптимизации
    
    gridParams = read_grid_params_from_config(config_path)
    
    # Создаем сетку
    
    params = [{'boosting_type':bt, 'colsample_bytree':cbt, 'learning_rate':lr, 'max_depth':md, 'min_child_samples':mcs, 'min_child_weight':mcw,
           'min_split_gain':msg, 'n_estimators':ne, 'num_leaves':nl, 'reg_alpha':l1, 'reg_lambda':l2, 'subsample':subs, 
           'subsample_for_bin':sfb, 'subsample_freq':sf} for bt in gridParams['boosting_type'] for cbt in gridParams['colsample_bytree'] 
          for lr in gridParams['learning_rate'] for md in  gridParams['max_depth'] for mcs in gridParams['min_child_samples'] 
          for mcw in gridParams['min_child_weight'] for msg in gridParams['min_split_gain'] for ne in gridParams['n_estimators']
          for nl in gridParams['num_leaves'] for l1 in gridParams['reg_alpha'] for l2 in gridParams['reg_lambda']
          for subs in gridParams['subsample'] for sfb in gridParams['subsample_for_bin'] for sf in gridParams['subsample_freq']]
    
    # Проводим опитимазацию по целевой метрике на валидационной выборке
    
    X_train, y_train = train_df[feat_cols[:-1]], train_df[feat_cols[-1]]
    
    best_mean_av_prec = 0
    best_params = params[0]
    
    print('Оптимизация')
    for param in tqdm(params, position=0):
        lgbmc = Model(param, random_state)
        
        lgbmc.train(X_train, y_train)
        
        mean_av_prec = 0
        total = 0
        
        for line in tqdm(val_df.values, position=0):
            
            user_id, bought_products, user_region, user_total_buy, user_dif_cat = line
            
            temp_df = all_prods.copy()
            
            temp_df['user_region'] = user_region
            temp_df['user_region'] = temp_df['user_region'].astype(CategoricalDtype(categories=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))
            temp_df['user_total_buy'] = user_total_buy
            temp_df['user_dif_cat'] = user_dif_cat
            
            temp_df['probs'] = lgbmc.predict_proba(temp_df[feat_cols[:-1]])[:, 1]
            
            sorted_recommended_products = temp_df['product_id'].values[np.argsort(temp_df['probs'].values)[::-1]]
            
            mean_av_prec += average_precision(bought_products, sorted_recommended_products, 10)
            total += 1
            
            mean_av_prec /= total
            
        if mean_av_prec > best_mean_av_prec:
            best_mean_av_prec = mean_av_prec
            best_params = param
            
            lgbmc.save_model('best_model.pkl')