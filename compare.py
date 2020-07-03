import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import pickle as pkl
from pandas.api.types import CategoricalDtype
from Model import Model
import sys
from optimization import average_precision
from scipy import stats

# Функция для подсчета доверительного интервала для оценки неопределенности модели
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return [m, h]

best_model = Model()
best_model.load_model('best_model.pkl')
best_params = best_model._get_params() # Считываем лучшие гиперпараметры с модели
random_states = [12, 4124, 10, 23, 4] # Различные инициализации модели для оценки неопределенности

feat_cols = ['user_region', 'product_category', 'price', 'asset', 'user_total_buy', 'user_dif_cat', 'product_total_buy', 'prod_dif_region', 'label'] # Колонки для обучения и метка

if __name__=='__main__':
    
    dir_path = str(sys.argv[1]) # # Путь к папке с обучающим, валидационным, тестовым и различными вспомогательными датасетами, которые получились в результате выполнения скрипта preprocess.py
    
    train_df = pd.read_csv(dir_path + '/train_dataset.csv')
    test_df = pd.read_csv(dir_path + '/test_df.csv')
    all_prods = pd.read_csv(dir_path + '/all_prods.csv')
    all_users = pd.read_csv(dir_path + '/all_users.csv')
    
    train_df['user_region'] = train_df['user_region'].astype(CategoricalDtype(categories=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))
    train_df['product_category'] = train_df['product_category'].astype('category')
    
    test_df['user_region'] = test_df['user_region'].astype(CategoricalDtype(categories=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))
    test_df['bought_products'] = test_df['bought_products'].apply(lambda x:np.fromstring(x.strip('[ ]'), dtype=int, sep=' '))
    
    all_prods['product_category'] = all_prods['product_category'].astype('category')
    
    all_users['user_region'] = all_users['user_region'].astype(CategoricalDtype(categories=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))
    
    X_train, y_train = train_df[feat_cols[:-1]], train_df[feat_cols[-1]]
    
    # Оценка неопределенности модели по целевой метрике на тестовой выборке
    
    mean_average_precision_lgbm = []
    for random_state in tqdm(random_states, position=0):
        
        lgbmc = Model(best_params, random_state)
        
        lgbmc.train(X_train, y_train)
        
        mean_av_prec = 0
        total = 0
        
        for line in tqdm(test_df.values, position=0):
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
            
    mean_average_precision_lgbm.append(mean_av_prec/total)
    
    print('\n')
    print('Значение целевой метрики для модели бустинга: ', mean_confidence_interval(mean_average_precision_lgbm))
    print('\n')
    
    # Рекомендуем 10 самых популярных продуктов среди всех покупателей
    
    recommended_prods = all_prods.sort_values('product_total_buy', ascending=False)['product_id'].values[:10]
    
    mean_average_precision_popular = 0
    total = 0
    
    for line in tqdm(test_df.values, position=0):
        user_id, bought_products, user_region, user_total_buy, user_dif_cat = line
        
        mean_average_precision_popular += average_precision(bought_products, recommended_prods, 10)
        total += 1
        
    print('\n')
    print('Значение целевой метрики для модели рекомендации 10 самых популярных товаров: ', mean_average_precision_popular)
    print('\n')
    
    # Рекомендуем 10 самых популярных продуктов среди покупателей из одного региона
    
    mean_average_precision_popreg = 0
    total = 0
    
    for line in tqdm(test_df.values, position=0):
        user_id, bought_products, user_region, user_total_buy, user_dif_cat = line
        
        temp_df = train_df[(train_df['label']==1) & (train_df['user_region'] == user_region)].copy()
        count_prod = temp_df.groupby('product_id')['product_id'].count().to_frame('count_prod').reset_index()
        
        recommended_prods = count_prod.sort_values('count_prod', ascending=False)['product_id'].values[:10]
        
        mean_average_precision_popreg += average_precision(bought_products, recommended_prods, 10)
        total += 1
        
    print('\n')
    print('Значение целевой метрики для модели рекомендации 10 самых популярных товаров среди пользователей из того же региона: ', mean_average_precision_popreg)
    print('\n')
    
    # Рекомендуем 10 самых популярных продуктов среди 10 похожих покупателей
    
    mean_average_precision_nn = 0
    total = 0

    for line in tqdm(test_df.values, position=0):

        user_id, bought_products, user_region, user_total_buy, user_dif_cat = line

        temp_df = train_df[(train_df['label'] == 1) & (train_df['user_id'] != user_id)]

        nn = NearestNeighbors(10)
        nn_df = temp_df.drop_duplicates('user_id')
        nn.fit(nn_df[['user_total_buy', 'user_dif_cat']].values)

        _, nearest_indexes = nn.kneighbors(np.reshape([user_total_buy, user_dif_cat], (1, 2)))

        closest_users_arr = nn_df.iloc[nearest_indexes[0], 0].values
        closest_users = temp_df[temp_df['user_id'].isin(closest_users_arr)]
        popular_products = closest_users.groupby('product_id')['product_id'].count().to_frame('popular_products').reset_index()
        recommended_prods = popular_products.sort_values('popular_products', ascending=False)['product_id'].values[:10]

        mean_average_precision_nn += average_precision(bought_products, recommended_prods, 10)
        total += 1

    print('\n')
    print('Значение целевой метрики для модели рекомендации 10 самых популярных товаров среди 10 похожих пользователей: ', mean_average_precision_nn)
    print('\n')
    
    # Рекомендуем 10 случайных продуктов
    
    np.random.seed(432)

    mean_average_precision_random = 0
    total = 0

    for line in tqdm(test_df.values, position=0):
        recommended_prods = np.random.choice(all_prods['product_id'].values, 10, replace=False)
        user_id, bought_products, user_region, user_total_buy, user_dif_cat = line

        mean_average_precision_random += average_precision(bought_products, recommended_prods, 10)
        total += 1

    print('\n')
    print('Значение целевой метрики для модели рекомендации 10 случайных товаров: ', mean_average_precision_random)
    print('\n')