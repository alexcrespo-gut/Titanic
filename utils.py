#!/usr/bin/python
# _*_coding:utf-8_*_

# Import modules
import re
import os
from collections import defaultdict
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import TransformerMixin, BaseEstimator


# Data import
def data_import(path):
    """ Imports train and test set to pandas dataframe."""
    train_path = os.path.join(path, "train.csv")
    test_path = os.path.join(path, "test.csv")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    return df_train, df_test


def get_title_dict(df):
    """
    Explain function
    :param df:
    :return:
    """
    tittle = defaultdict(lambda: 0)
    for name in df.Name:
        match = re.search(r', ([\w]*)\.', name)
        if match:
            tittle[match.group(1)] += 1
    return dict(tittle)


def get_cabin_dict(df):
    """
    Explain function
    :param df:
    :return:
    """
    cabin_dict = defaultdict(lambda: 0)
    for cabin in df.Cabin:
        # Test if it's equal to himself for nan check
        if cabin == cabin:
            match = re.search(r"[A-Z]", cabin)
            if match:
                cabin_dict[match.group(0)] += 1
        else:
            cabin_dict['None'] += 1

    return dict(cabin_dict)


def get_ticket_dict(df):
    ticket_dict = defaultdict(lambda: 0)
    for ticket in df.Ticket:
        ticket = ticket.split(" ")
        if len(ticket) > 1:
            ticket_dict[ticket[0]] += 1
        else:
            ticket_dict['none'] += 1
    return dict(ticket_dict)


def add_title(df, search_for):
    """
    Creates a new column with the titles to search for. The observations which name's title is not included in the list
    are labeled as others.
    :param dumies:
    :param search_for:
    :param df:
    :return:
    """
    # Copy the dataset.
    new_df = df.copy()
    # Create new column with the title in the name
    new_df['Title'] = df.Name.str.extract(r', (\w*)\.')
    # Create regex to find title
    regex = []
    for title in search_for:
        regex.append(r', ' + title + r'\.')
    # Find the which names contain any of the titles to search for.
    title_in_list = df.Name.str.contains('|'.join(regex))
    # Replace the titles that are not in list for 'Other'.
    new_df.loc[~title_in_list, 'Title'] = 'Other'

    return new_df


def impute_average(df_train, variable_list, df_test):
    """

    :param df_test:
    :param variable_list:
    :param df_train:
    :return:
    """
    # Define the list to save the impute
    imp_list = []
    # Repeat for each variable
    for variable in variable_list:
        # Check if is numeric
        if df_train[variable].dtypes in {int, float}:
            imp = SimpleImputer(missing_values=np.NaN, strategy='mean')
        else:
            imp = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
        # Fit and transform
        imp.fit(df_train[[variable]])
        SimpleImputer()
        df_train[variable] = imp.transform(df_train[[variable]]).ravel()
        df_test[variable] = imp.transform(df_test[[variable]]).ravel()
    # Return df or df+imp
    return df_train, df_test


# Get categorical features
def categorical(df, remove_ov=True):
    """ Return a list of the columns variable names which types is object or categorical"""
    object_features = df.loc[:, df.dtypes == 'object'].columns.tolist()
    categorical_features = df.loc[:, df.dtypes == 'category'].columns.tolist()
    features = list(set(object_features + categorical_features))
    if remove_ov:
        try:
            features.remove("Survived")
        except ValueError:
            None

    return features


# Get numerical features
def numeric(df, remove_ov=True):
    """ Return a list of the column variable names which types is float or integer"""
    float_features = df.loc[:, df.dtypes == 'float'].columns.tolist()
    integer_features = df.loc[:, df.dtypes == 'int'].columns.tolist()
    numeric_features = list(set(float_features + integer_features))
    if remove_ov:
        try:
            numeric_features.remove("Survived")
        except ValueError:
            None

    return numeric_features


class Dummify(BaseEstimator, TransformerMixin):
    """
    dummify variables to further analysis.
    """
    def __init__(self, variables, to_drop=""):
        self.variables = variables
        self.to_drop = to_drop
        self.dummy_columns = []
        self.dummy_features_ = []

    def fit(self, df, y=None):
        # Gets the dummy variables
        for v in self.variables:
            self.dummy_columns.append(pd.get_dummies(df[v], prefix=v))
        return self

    def transform(self, df, y=None):
        # Make hard copy to not modify original
        df = df.copy()
        # For each variable to transform
        for n, v in enumerate(self.variables):
            # For each dummy variable of variables to transform.
            for i in self.dummy_columns[n]:
                # Find which observations are equal to the tittle without the prefix.
                df[i] = df[v].apply(str) == i[len(v)+1:]
                # Change datatype
                df[i] = df[i].astype('int').astype('category')
                # Save the variable name
                self.dummy_features_.append(i)
            # Drop original variable and one of the dummy variables if indicated.
            if self.to_drop[n]:
                df.drop(labels=[v, v + "_" + str(self.to_drop[n])], axis=1, inplace=True)
                self.dummy_features_.remove(v + "_" + str(self.to_drop[n]))
            else:
                df.drop(labels=v, axis=1, inplace=True)
        return df


def bin_histogram (modified_df, v_to_bin):
    """For each variable in the list , prints the corresponding histogram for survived and not survived passengers"""
    for variable in v_to_bin:
        # Remove Nas
        df = modified_df[modified_df[variable].notnull()]
        # Create surv filter
        hist_filter = df["Survived"] == 1
        # Create Histogram
        plt.hist([df[variable][hist_filter], df[variable][~hist_filter]],
                 stacked=True, label=['Survived', 'Not Survived'], color=['g', 'r'])
        plt.legend()
        # Save and reset fig
        plt.savefig(variable+"_histogram")
        plt.clf()

