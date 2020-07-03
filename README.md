# Sber_RecSys
## Тестовое задание по рекомендательным системам от Сбербанка
Archive - папка с исходными данными

preprocessing.py - скрипт для предобработки исходных данных, которые лежат в папке Archive, принимает на вход путь до папки с исходными данными, а также число N, количество ближайших соседей, учитывающихся при заполнении пропусков в данных и для решения проблемы холодного старта.

Generated_Data - предобработанные данные, возникшие в результате работы скрипта preprocessing.py

optimization.py - скрипт для оптимизации гиперпараметров модели, принимает на вход путь к обучающей и валидационной выборкам, собранным скриптом preprocessing.py, а также путь до конфигурационного файла с значениями оптимизируемых гиперпараметров.

opt_conf.ini - конфигурационный файл с сеткой значений оптимизируемых гиперпараметров модели.

Model.py - класс модели бустинга lgbm, созданный для удобства объявления и обучения.

best_model.pkl - объект класса Model с наилучшими параметрами, подобранными в скрипте optimization.py.

compare.py - сравнение различных подходов с моделью по целевой метрике на тестовой выборке.

Project.ipynb - Jupyter Notebook с анализом данных.

