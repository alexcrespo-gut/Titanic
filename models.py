#!/usr/bin/python
# _*_coding:utf-8_*_


import warnings
import re
from collections import defaultdict

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV

from utils import Dummify
warnings.simplefilter(action='ignore', category=FutureWarning)


def split_train_test(df):
    """ Given a pandas dataframe splits it in train/test arrays."""
    y = df['Survived']
    x = df.loc[:, df.columns != 'Survived']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=123)
    return x_train, x_test, y_train, y_test


def sort_feature_imp(importances, features):
    """ Given a list of importances and the features it returns tuples with ordered feature importance"""
    # Join dummified variables importances in dict
    importances_dict = defaultdict(lambda: 0)
    for n, variable in enumerate(features):
        match = re.search(r'(\w*)_', variable)
        if match:
            importances_dict[match.group(1)] += importances[n]
        else:
            importances_dict[variable] += importances[n]

    # Print results sorted
    importance_tuple = sorted(importances_dict.items(), key=lambda item: item[1], reverse=True)
    print("Feature ranking:")
    for n, t in enumerate(importance_tuple):
        print("{}.  {} ({:.4f})".format(n+1, t[0], t[1]))

    return importance_tuple


def get_feature_drop_list(df):
    """ Gets features list and drop list for Dummify"""
    class_drop = {"Pclass": "1", "Cabin": "False", "Title": "Other", "Fare_binned": "Cheap", "Age_binned": "Elder",
                  "Embarked": "Q", "SibSp": "5", "Parch": "1"}
    features_list = [feature for feature in df.columns if feature != 'Survived']
    drop_list = [class_drop[feature] for feature in features_list]
    return features_list, drop_list


def cv_model(df, model):
    """ Calculate cv forest for the given dataset"""
    # Define pipe features
    features_list, drop_list = get_feature_drop_list(df)
    # Define X and Y
    y = df['Survived']
    X = df.loc[:, df.columns != 'Survived']
    # Define CV and model
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
    rfe_model = model
    pipe = make_pipeline(Dummify(features_list, drop_list), rfe_model)
    # Calculate score
    scores = cross_val_score(pipe, X, y, cv=kfold)

    return scores.mean()


def forest_cv_rfe(df, model):
    best_acc = cv_model(df, model)
    while len(df.columns) > 2:
        # Calculate accuraccy of the model with all variables
        feature_list = [feature for feature in df.columns if feature != 'Survived']
        print("The accuracy of the model with {} features is {:.3f}".format(len(feature_list), best_acc))
        # Define variables to drop feature
        remove_feature = None
        # Remove one variable
        for feature in feature_list:
            acc = cv_model(df.drop([feature], axis=1), model)
            print("\t- The accuracy removing {} is {}.".format(feature, acc))
            if acc > best_acc:
                best_acc = acc
                remove_feature = feature
        # Drop worst feature
        if remove_feature:
            df = df.drop([remove_feature], axis=1)
            print("\nFeature {} has been removed.".format(remove_feature))
        else:
            print("\nThere is no better model removing more variables. The selected variables are {}."
                  .format(','.join(feature_list)))
            return feature_list
    return feature_list


def gs_cv(df, model, param_grid):
    """ Returns the best model"""
    # Define X, y
    X_train, X_test, y_train, y_test = split_train_test(df)

    # Define pipe
    feature_list, drop_list = get_feature_drop_list(df)
    pipe = make_pipeline(Dummify(feature_list, drop_list), model)

    # Grid Search Set up
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    if param_grid:
        gs = GridSearchCV(pipe, param_grid, n_jobs=-1, cv=kfold)
    else:
        gs = GridSearchCV(pipe, n_jobs=-1, cv=kfold)
    # Grid Search fit and predict
    gs.fit(X_train, y_train)
    # Verbosity
    # df_cv_scores = pd.DataFrame(gs.cv_results_).sort_values(by='rank_test_score')
    # print(df_cv_scores.loc[0:10, ['params', 'mean_test_score', 'std_test_score']] .to_string())
    return gs.score(X_test, y_test), gs.best_params_


def model_tuning(df, models_dict):
    print("The accuracy on the test for each model is: ")
    for model in models_dict.keys():
        # Calculate the best hyperparameters for each model and compare the models with a validation set.
        gs = gs_cv(df, models_dict[model]['model'], models_dict[model]['param_grid'])
        # Print and save results
        print('\t - For {} : {:.4f} '.format(model, gs[0]))
        models_dict[model]['score'] = gs[0]
        models_dict[model]['best_params'] = gs[1]
    print("")
    return models_dict


def train_and_predict(best_model, df_train, df_test, models_dict, return_imp=True):
    """
    Train the best model with whole dataset and predicts the result for the test set. If importances is True, it returns
    the feature importance for each variable.
    :param return_imp:
    :param best_model:
    :param df_train:
    :param df_test:
    :param models_dict:
    :return:
    """
    # Define train X y
    X_train = df_train.loc[:, df_train.columns != 'Survived']
    y_train = df_train['Survived']

    # Set up model
    model = models_dict[best_model]['model']

    # Set up pipe
    pipe = make_pipeline(Dummify(*get_feature_drop_list(df_train)), model)
    pipe.set_params(**models_dict[best_model]['best_params'])

    # Fit the model
    pipe.fit(X_train, y_train)

    if return_imp:
        # Define importances and features
        importances = pipe.steps[1][1].feature_importances_
        print(importances)
        features = pipe.steps[0][1].dummy_features_
        print(features)
        # Show feature importance
        sort_feature_imp(importances, features)

    return pipe.predict(df_test)


#
# def pipe_acc_test(df, estimators=[]):
#     # Define the Test and train
#     X_train, X_test, y_train, y_test = split_train_test(df)
#     # Define the models to analyze
#     models_list = [DecisionTreeClassifier(), RandomForestClassifier(),
#                    GradientBoostingClassifier(), KNeighborsClassifier(), SVC()]
#     pipes_dict = {0: "Decision Tree", 1: "Random Forest", 2: "GBT", 3: "KNN", 4: "SCV"}
#     pipes_list = []
#
#     # Join the models with the previous estimators
#     if type(estimators) == list:
#         for model in models_list:
#             pipes_list.append(make_pipeline(*estimators, model))
#     else:
#         for model in models_list:
#             pipes_list.append(make_pipeline(estimators, model))
#
#     # Create a variable to store best accuraccy result
#     best_accuracy = 0
#
#     # Train and test the pipes
#     for n, pipe in enumerate(pipes_list):
#         pipe.fit(X_train, y_train)
#         y_pred = pipe.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)
#         print(" The accuracy for {} model is {:.3f}.".format(pipes_dict[n], accuracy))
#         # Save the best result
#         if accuracy > best_accuracy:
#             best_accuracy = accuracy
#             best_pipe = pipe
#
#     return best_pipe

#
# def cv_acc(df, pipe):
#     # Define X and Y
#     y = df['Survived']
#     X = df.loc[:, df.columns != 'Survived']
#
#     # Define the models to analyze
#     models_list = [DecisionTreeClassifier(random_state=123), RandomForestClassifier(random_state=123),
#                    GradientBoostingClassifier(random_state=123), KNeighborsClassifier(), SVC(random_state=123)]
#     pipes_dict = {0: "Decision Tree", 1: "Random Forest", 2: "GBT", 3: "KNN", 4: "SCV"}
#     pipes_list = []
#
#     # Join pipe with the model
#     for model in models_list:
#         pipes_list.append(make_pipeline(pipe, model))
#
#     # Train and test the pipes
#     for n, pipe in enumerate(pipes_list):
#         scores = cross_val_score(pipe, X, y, cv=10)
#         print("The accuracy for {} model is {:.3f} with a standard deviation of {:.3f}"
#               .format(pipes_dict[n], scores.mean(), scores.std()))
#     return None


# def accuracy_test(df, numeric_estimators, categorical_estimators, cross_val=False):
#     # Add modifications to y
#     categorical_features = categorical(df)
#     numeric_features = numeric(df)
#     ct = ColumnTransformer([("Num", make_pipeline(*numeric_estimators), numeric_features),
#                             ("Cat", make_pipeline(*categorical_estimators), categorical_features)])
#     if cross_val:
#         cv_acc(df, ct)
#     else:
#         pipe_acc_test(df, ct)
#     return ct