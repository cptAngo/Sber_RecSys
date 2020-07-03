import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from pandas.api.types import CategoricalDtype
import sys
import os

import gc
gc.enable()

np.random.seed(432) # Для повторяемости результатов


dir_path = str(sys.argv[1]) # Путь к папке с исходными данными
N_N = int(sys.argv[2]) # Количество ближайших соседей, учитывающихся при всех алгоритмах заполнения пропусков в данных

try:
    os.mkdir('Generated_Data')
except:
    pass

if __name__=='__main__':

    # Считываем исходные данные, в процессе первичного анализа было выявлено, что не все колонки в данных несут какой-либо смысл, поэтому считываем сразу только полезные

    user_region = pd.read_csv(dir_path + '/user_region.csv', usecols=['row', 'col'])
    user_age = pd.read_csv(dir_path + '/user_age.csv', usecols=['row'])
    item_subclass = pd.read_csv(dir_path + '/item_subclass.csv', usecols=['row', 'col'])
    item_price = pd.read_csv(dir_path + '/item_price.csv', usecols=['row', 'data'])
    item_asset = pd.read_csv(dir_path + '/item_asset.csv', usecols=['row', 'data'])
    interactions = pd.read_csv(dir_path+ '/interactions.csv')

    # Проверка на пустые значения

    assert user_region.isnull().values.any() == False
    assert user_age.isnull().values.any() == False
    assert item_subclass.isnull().values.any() == False
    assert item_price.isnull().values.any() == False
    assert item_asset.isnull().values.any() == False
    assert interactions.isnull().values.any() == False

    # Собираем датасет для продуктов

    item_subclass.rename(columns={'col':'product_category'}, inplace=True)
    item_subclass['product_category'] = item_subclass['product_category'].astype('category')

    item_price.rename(columns={'data':'price'}, inplace=True)

    item_asset.rename(columns={'data':'asset'}, inplace=True)

    prod_feats = item_subclass.merge(item_price, how='outer', on='row').merge(item_asset, how='outer', on='row')
    prod_feats.rename(columns={'row':'product_id'}, inplace=True)

    del [item_price, item_asset]
    gc.collect()

    # Заполняем пропуски в данных для продуктов на основе алгоритма ближайших соседей: ищем похожие продукты из той же категории с точки зрения ненулевых признаков, затем заполняем пропуски усреднением соответствующих значений признаков у соседей

    prod_feats_without_nan = prod_feats[(~prod_feats['price'].isna()) & (~prod_feats['asset'].isna())]
    
    print('Заполняем первичные пропуски для продуктов')

    for ind in tqdm(prod_feats[prod_feats['price'].isna()].index, position=0):
        categ = prod_feats.loc[ind, 'product_category']
        asset = prod_feats.loc[ind, 'asset']

        temp_df = prod_feats_without_nan[(prod_feats_without_nan['product_category'] == categ)]

        n_neighbors = min(N_N, temp_df.shape[0])

        nn = NearestNeighbors(n_neighbors)
        nn.fit(temp_df['asset'].values.reshape(-1,1))

        _, neighbors_inds = nn.kneighbors(np.reshape([asset], (1, 1)))

        prod_feats.loc[ind, 'price'] = temp_df.values[neighbors_inds, -2].mean()

    for ind in tqdm(prod_feats[prod_feats['asset'].isna()].index, position=0):
        categ = prod_feats.loc[ind, 'product_category']
        price = prod_feats.loc[ind, 'price']

        temp_df = prod_feats_without_nan[(prod_feats_without_nan['product_category'] == categ)]

        n_neighbors = min(N_N, temp_df.shape[0])

        nn = NearestNeighbors(n_neighbors)
        nn.fit(temp_df['price'].values.reshape(-1,1))

        _, neighbors_inds = nn.kneighbors(np.reshape([price], (1, 1)))

        prod_feats.loc[ind, 'asset'] = temp_df.values[neighbors_inds, -1].mean()

    # Собираем датасет для пользователей

    user_region.rename(columns={'col':'user_region'}, inplace=True)

    user_feats = user_age.merge(user_region, how='outer', on='row')
    user_feats.rename(columns={'row':'user_id'}, inplace=True)

    user_feats['user_region'].fillna(1, inplace=True) # Наивно относим всех пользователей с неизвестным регионом к новому региону с номером 1
    user_feats['user_region'] = user_feats['user_region'].astype('category')

    del [user_region, user_age]
    gc.collect()

    # Выделяем обучающие, валидационные и тестовые заказы в соотношении 70/15/15

    interactions.rename(columns={'row':'user_id', 'col':'product_id', 'data':'label'}, inplace=True)


    train_indexes = np.random.choice(interactions.index, int(len(interactions.index)*0.7), replace=False)
    val_train_indexes = list(filter(lambda x: x not in train_indexes, interactions.index))

    val_indexes = val_train_indexes[:int(len(val_train_indexes)*0.5)]
    test_indexes = val_train_indexes[int(len(val_train_indexes)*0.5):]

    interactions_train = interactions.loc[train_indexes].reset_index(drop=True)
    interactions_val = interactions.loc[val_indexes].reset_index(drop=True)
    interactions_test = interactions.loc[test_indexes].reset_index(drop=True)

    # Генерируем признаки на основе обучающей выборки

    user_prod_train = interactions_train.merge(user_feats, how='inner', on='user_id').merge(prod_feats, how='inner', on='product_id')

    # Признаки пользователя

    user_total_buy = user_prod_train.groupby('user_id')['user_id'].count().to_frame('user_total_buy') # Полное число покупок пользователя
    user_total_buy = user_total_buy.reset_index()

    user_dif_cat = user_prod_train.groupby('user_id')['product_category'].nunique().to_frame('user_dif_cat') # Число различных категорий, которые покупал пользователь
    user_dif_cat = user_dif_cat.reset_index()

    # Холодный старт для пользователей

    all_users = user_feats.merge(user_total_buy, how='outer', on='user_id').merge(user_dif_cat, how='outer', on='user_id')
    all_users_without_nan = all_users[(~all_users['user_total_buy'].isna()) & (~all_users['user_dif_cat'].isna())]

    # Заполняем пропуски для некоторых пользователей путем усреднения значений соответствующих признаков пользователей из того же региона
    
    print('Заполняем пропуски для пользователей')
    
    for ind in tqdm(all_users[all_users['user_total_buy'].isna()].index, position=0):
        region = all_users.loc[ind, 'user_region']

        temp_df = all_users_without_nan[(all_users_without_nan['user_region'] == region)]

        all_users.loc[ind, 'user_total_buy'] = int(temp_df.loc[:, 'user_total_buy'].mean())
        all_users.loc[ind, 'user_dif_cat'] = int(temp_df.loc[:, 'user_dif_cat'].mean())

    for ind in tqdm(all_users[all_users['user_dif_cat'].isna()].index, position=0):
        region = all_users.loc[ind, 'user_region']

        temp_df = all_users_without_nan[(all_users_without_nan['user_region'] == region)]

        all_users.loc[ind, 'user_dif_cat'] = int(temp_df.loc[:, 'user_dif_cat'].mean())

    all_users.drop_duplicates(subset=['user_id'], inplace=True)

    all_users.to_csv('Generated_Data/all_users.csv', index=False)

    # Признаки продукта

    product_total_buy = user_prod_train.groupby('product_id')['user_id'].count().to_frame('product_total_buy') # Полное число покупок продукта
    product_total_buy = product_total_buy.reset_index()

    prod_dif_region = user_prod_train.groupby('product_id')['user_region'].nunique().to_frame('prod_dif_region') # Число различных регионов, в которых покупали данный продукт
    prod_dif_region = prod_dif_region.reset_index()

    # Холодный старт для продуктов

    all_prods = prod_feats.merge(product_total_buy, how='outer', on='product_id').merge(prod_dif_region, how='outer', on='product_id')
    all_prod_without_nan = all_prods[(~all_prods['product_total_buy'].isna()) & (~all_prods['prod_dif_region'].isna())]

    # Заполняем пропуски для некоторых продуктов путем усреднения значений соответствующих признаков продуктов из той же категории
    
    print('Заполняем пропуски для сгенерированных признаков продуктов')
    
    for ind in tqdm(all_prods[all_prods['product_total_buy'].isna()].index, position=0):
        categ = all_prods.loc[ind, 'product_category']
        price, asset = all_prods.loc[ind, 'price'], all_prods.loc[ind, 'asset']

        temp_df = all_prod_without_nan[(all_prod_without_nan['product_category'] == categ)]
        if temp_df.shape[0] == 0:
            temp_df = all_prod_without_nan

        n_neighbors = min(N_N, temp_df.shape[0])

        nn = NearestNeighbors(n_neighbors)
        nn.fit(temp_df[['price', 'asset']].values)

        _, neighbors_inds = nn.kneighbors(np.reshape([price, asset], (1, 2)))

        all_prods.loc[ind, 'product_total_buy'] = int(temp_df.values[neighbors_inds, -2].mean())
        all_prods.loc[ind, 'prod_dif_region'] = int(temp_df.values[neighbors_inds, -1].mean())

    all_prods.drop_duplicates(subset=['product_id'], inplace=True)

    all_prods.to_csv('Generated_Data/all_prods.csv', index=False)

    # Соединяем все признаки с таблицей взаимодействия пользователей с продуктами

    user_prod_train = interactions_train.merge(all_users, how='inner', on='user_id').merge(all_prods, how='inner', on='product_id')
    
    print('Собираем датасеты для валидации и теста')

    val_arr = []
    for user_id, group_u in tqdm(interactions_val.groupby('user_id'), position=0):
        val_arr.append([user_id, group_u['product_id'].values])

    val_df = pd.DataFrame(val_arr, columns=['user_id', 'bought_products']).merge(all_users, how='inner', on='user_id')

    test_arr = []
    for user_id, group_u in tqdm(interactions_test.groupby('user_id'), position=0):
        test_arr.append([user_id, group_u['product_id'].values])

    test_df = pd.DataFrame(test_arr, columns=['user_id', 'bought_products']).merge(all_users, how='inner', on='user_id')

    val_df.to_csv('Generated_Data/val_df.csv', index=False)
    test_df.to_csv('Generated_Data/test_df.csv', index=False)

    del [user_feats, prod_feats, val_df, test_df]
    gc.collect()

    # Генерируем отрицательные классы: для каждого пользователя генерируем столько же отрицательных примеров, сколько у него положительных в обучающей выборке
    
    print('Генерируем отрицательные классы для обучающей выборки')

    negative_samples = []
    for user_id, group_u in tqdm(user_prod_train.groupby('user_id'), position=0):

        product_list = list(set(group_u['product_id']))

        target_products = list(set(user_prod_train[(user_prod_train['user_id'] != user_id) & (~user_prod_train['product_id'].isin(product_list))]['product_id']))

        num_to_extract = min(len(product_list), len(target_products))

        negative_products = np.random.choice(target_products, num_to_extract, replace=False)

        for product in negative_products:
            negative_samples.append([user_id, product])

    negative_user_prod_train = pd.DataFrame(negative_samples, columns=['user_id', 'product_id'])
    negative_user_prod_train['label'] = 0

    negative_user_prod_train = negative_user_prod_train.merge(all_users, how='inner', on='user_id').merge(all_prods, how='inner', on='product_id')

    df = pd.concat((user_prod_train, negative_user_prod_train), axis=0).sample(frac=1)

    df.to_csv('Generated_Data/train_dataset.csv', index=False)
    
    print('Предобработка завершена')
