#!/usr/bin/python
# _*_coding:utf-8_*_

# Import modules
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import f_oneway, shapiro, chi2_contingency
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Custom modules
from utils import numeric, categorical
from models import forest_cv_rfe


# Normality test
def normality_test(df, variable, alpha=0.03):
    """ Calculate Shapiro - Wilks normality test."""
    stat, p = shapiro(df.loc[:, [variable]])
    if p < alpha:
        print("\t- {} looks Gaussian, fail to reject H0.".format(variable))
    else:
        print("\t- {} doesn't look Gaussian, reject H0.".format(variable))
    return None


# Anova test
def anova_test2(df, v_cat, v_num):
    # Remove nulls
    df = df.loc[df[v_num].notnull(), [v_cat, v_num]].copy()
    # Calculate the classes
    classes = set(df[v_cat])
    # Create the lists
    obs = [list(df.loc[df[v_cat] == i, v_num]) for i in classes]

    # Calculate anova
    try:
        anova = f_oneway(*obs)[1]
    except TypeError: # Error: df slice with no nulls values there is only on class of the categorical variable.
        anova = 0
    return anova


def chisquare_test(df, object_variable='Survived', verbosity=False):
    # Define the numerical columns to study
    cols_list = categorical(df, remove_ov=False)
    # Calculate chitest for each categorical variable
    if verbosity:
        print("Chi-square test is performed with {} for each other categorical variables.".format(object_variable))
        print("The results obtained are the following:")
    # Result Dictionary
    result_dict = {}
    # For each other colum in dataset calculate the chi-square test
    for col in cols_list:
        # Get the survived and not survived
        freq = df.groupby([col, object_variable]).size().unstack(fill_value=0).stack()
        # Calculate the matrix of frequencies for each class:
        v_index = set(df[object_variable])
        total_obs = [list(freq[freq.index.get_level_values(1) == i]) for i in v_index]
        # Chisquare test
        crosstab = np.array(total_obs)
        chi_test = chi2_contingency(crosstab)
        # V Cramer
        obs = np.sum(crosstab)
        mini = min(crosstab.shape)-1
        # Return results
        result_dict[col] = chi_test[0]/(obs*mini)
        if verbosity:
            print("\t- {}: The p-value is {}".format(col, chi_test[0]/(obs*mini)))
    return result_dict


def chisquare_matrix(df):
    # Define empty dictionary
    matrix = {}
    # Determine the categorical variables of the dataframe
    categorical_variables = categorical(df, remove_ov=False)
    # For each variable calculate the Cramer's V test and add it to dictionary
    for column in categorical_variables:
        matrix[column] = chisquare_test(df, column)
    # Convert dictionary of all test into a dataframe
    matrix = pd.DataFrame.from_dict(matrix)
    # Return the matrix with all Cramers V test ordered.
    return matrix[matrix.index]


def anova_matrix(modified_df):
    """Given a dataframe calculates all the anova test for numerical-categorical variables combinations"""
    matrix = defaultdict(dict)
    for c_variable in categorical(modified_df, remove_ov=False):
        for n_variable in numeric(modified_df):
            # Calculate anova
            anova = anova_test2(modified_df, c_variable, n_variable)
            # Add value to the dictionary
            matrix[c_variable][n_variable] = anova
    return pd.DataFrame.from_dict(dict(matrix))


def feature_selection(df):
    """ Finds the relevant features for further study"""
    print("\nFEATURE SELECTION:\n")

    # Anova Test: Numerical - Categorical Anova
    print("Shapiro-Wilk Test is performed on numerical variables to check Gaussian distribution.")
    normality_test(df, "Fare")
    normality_test(df[df.Age.notnull()], "Age")
    print("")
    print("Anova test is performed with the numerical variables. The results obtained are the following:")
    print(anova_matrix(df).to_string())
    print("")

    # Cramers V Test Matrix
    print("Because of the large sample, chi-square test is not adequate as it ....")
    print("Cramer's V Test is performed with categorical variables to identify if there is relation with the object"
          "variable and study collinearity between predicted variables.")
    print(chisquare_matrix(df).to_string())
    print("")

    # Correlation matrix - Numerical - Numerical
    print("To study collinearity between numerical variables, a correlation analysis is performed.")
    print(df.corr())
    print("")

    # Drop non relevant features
    non_relevant = ['PassengerId', 'Age']
    print("{} are considered non relevant and dropped from the data set.".format(','.join(non_relevant)))
    df.drop(non_relevant, axis=1, inplace=True)

    # Drop Collinear variables
    collinear = ['Sex', 'Fare']
    print("{} are considered collinear, we drop them because  lower Cramers V coefficient.".format(','.join(collinear)))
    df.drop(collinear, axis=1, inplace=True)
    print("")

    # RFE
    print('Recursive Feature Elimination is performed to determine the best features:')
    # Define dictionary for removing variables
    features = forest_cv_rfe(df, RandomForestClassifier(random_state=123))
    print("")
    return features




# # Anova test
# def anova_test(df, verbosity=False):
#     # Define the numerical columns to study
#     cols_list = numeric(df)
#     # Calculate boolean matrix for survived passengers
#     survived = df.Survived == 0
#     # Anova
#     for col in cols_list:
#         # Remove na's
#         no_null_df = df.loc[df[col].notnull(), :]
#         # Calculate anova
#         anova = f_oneway(np.array(no_null_df.loc[survived, col]), np.array(no_null_df.loc[~survived, col]))
#         if verbosity:
#             print("\t- {}: The p-value is {}".format(col, anova[1]))
#     return None
#