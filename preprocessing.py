#!/usr/bin/python
# _*_coding:utf-8_*_

# Import modules
import pandas as pd
import numpy as np

from utils import get_title_dict, get_cabin_dict, get_ticket_dict, add_title, impute_average,  bin_histogram



# Data review
def data_review(df):
    """ Prints summary of data review."""
    # Full data review
    print("\nDATA REVIEW:\n")
    print("There are a total of {} variables and {} observation in the dataset.\n"
          .format(len(df.columns), df.PassengerId.count()))
    print("The first columns of the training dataset are the following:")
    print(df.head(10).to_string())
    print("")

    # Social Position - Make it a class and transform also df_test !
    print("From the variable name it can be analyzed their role in the society.")
    print(get_title_dict(df))
    print("")

    # Cabin - Make it a class and transform also df_test ! or maybe not needed ...
    print("From the Cabin variable it can be analyzed which passengers got Cabins and the letters")
    print(get_cabin_dict(df))
    print("")

    # Find Out more about tickets
    print("Some tickets contain additional information from the number. Check if there is anything relevant")
    print(get_ticket_dict(df))
    print("")

    # Survived not survived
    print("In the training data set, the {:.2f} % of the people survived. There fore the dataset is imbalanced.\n"
          .format(df.Survived[df.Survived == 1].count() / df.Survived.count() * 100))

    # Check for null values
    print("Variables with null values are the following:")
    print(df.loc[:, df.isnull().sum() != 0].isnull().sum())
    print("")

    return None


def pre_processing(df_train, df_test):
    """ Basic data preprocessing for further analysis."""
    print("The following modifications have been done to the original dataset:")

    print("\t- Variable Title is added to the dataset")
    search_for = ['Mr', 'Mrs', 'Miss', 'Master', 'Rev', 'Dr']
    df_train = add_title(df_train, search_for)
    df_test = add_title(df_test, search_for)

    print("\t- Variable cabin is transformed to 1/0")
    df_train.Cabin = df_train.Cabin.notnull().astype(object)
    df_test.Cabin = df_test.Cabin.notnull().astype(object)

    print("\t- Variable ticket and name are dropped from the dataset")
    df_train.drop(['Ticket', 'Name'], axis=1, inplace=True)
    df_test.drop(['Ticket', 'Name'], axis=1, inplace=True)

    v_to_bin = ['Fare', 'Age']
    print("\t - Variables {} are binned into x categories".format(','.join(v_to_bin)))
    bin_histogram(df_train, v_to_bin)
    for n, variable in enumerate(v_to_bin):
        # Define the bins
        bin_cut = [[-np.inf, 50, 100, np.inf], [-np.inf, 7, 16, 60, np.inf]]
        label = [['Cheap', 'Normal', 'Expensive'], ['Kid', 'Teen', 'Adult', 'Elder']]
        # Bin age and Fare
        df_train[variable + '_binned'] = pd.cut(df_train[variable], bins=bin_cut[n], labels=label[n])
        df_test[variable + '_binned'] = pd.cut(df_test[variable], bins=bin_cut[n], labels=label[n])

    print("\t- Nan for age and embarked are estimated. The estimation strategy is to assign the most frequent value to "
          "the missing values\n")
    # Make Fit transform !
    df_train, df_test = impute_average(df_train, ['Embarked', 'Age_binned'], df_test=df_test)

    # Redefine variables
    v_to_cat = ['Pclass', 'SibSp', 'Parch', 'Sex', 'Cabin', 'Embarked', 'Age_binned']
    df_train[v_to_cat+['Survived']] = df_train[v_to_cat+['Survived']].astype('category')
    df_test[v_to_cat] = df_test[v_to_cat].astype('category')
    print("\t- Variable {} are defined as object and considered as categorical".format(', '.join(v_to_cat)))

    return df_train, df_test