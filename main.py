#!/usr/bin/python
# _*_coding:utf-8_*_

# Import modules
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
# Custom modules
from utils import data_import
from models import model_tuning, train_and_predict
from feature_selection import feature_selection
from preprocessing import data_review, pre_processing


if __name__ == '__main__':
    # Import Data
    path = "data/"
    df_train, df_test = data_import(path)

    # Data review
    data_review(df_train)

    # Data preprocessing
    df_train, df_test = pre_processing(df_train, df_test)

    # Feature Selection
    features = feature_selection(df_train)

    # Model training: define dataset
    df_train = df_train.loc[:, features + ['Survived']]
    df_test = df_test.loc[:, features]

    # Model training: define models
    models_dict = {"Decision tree": {'model': DecisionTreeClassifier(random_state=123),
                                     'param_grid': {'decisiontreeclassifier__min_samples_split': [2, 10, 25, 40],
                                                    'decisiontreeclassifier__max_depth': [5, 10, 20],
                                                    'decisiontreeclassifier__min_samples_leaf': [2, 5, 10, 20]},
                                     'score': None,
                                     'best_params': None},
                   'Random forest': {'model': RandomForestClassifier(random_state=123),
                                     'param_grid': {'randomforestclassifier__n_estimators': [50, 100, 500],
                                                    'randomforestclassifier__max_depth': [5, 7, 10, 30, 100]},
                                     'score': None,
                                     'best_params': None},
                   'Gradient Boosting': {'model': GradientBoostingClassifier(random_state=123),
                                         'param_grid': {'gradientboostingclassifier__learning_rate': [0.1, 0.01, 0.001],
                                                        'gradientboostingclassifier__n_estimators': [100, 200, 500],
                                                        'gradientboostingclassifier__subsample':  [0.7, 0.8, 0.9, 1]},
                                         'score': None,
                                         'best_params': None}}

    # Model training: Split df( Train set GridSearchCV to find best hyper-parameters - validation set to compare models)
    models_dict = model_tuning(df_train, models_dict)

    # Model training: select model
    print("The model Selected is .... because ...")
    best_model = 'Random forest'

    # Model training: train and predict
    y_pred = train_and_predict(best_model, df_train, df_test, models_dict, True)

    # Save results




