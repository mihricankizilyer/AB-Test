import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, pearsonr,\
    spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' %x)

df_control = pd.read_excel('datasets/5W/ab_testing.xlsx', sheet_name='Control Group')
df_control = df_control[["Impression","Click","Purchase","Earning"]]

df_test = pd.read_excel('datasets/5W/ab_testing.xlsx', sheet_name='Test Group')
df_test = df_test[["Impression","Click","Purchase","Earning"]]


def check_df(dataframe, head = 5):
    print("######### Shape #########")
    print(dataframe.shape)
    print("######### Types #########")
    print(dataframe.dtypes)
    print("######### Head #########")
    print(dataframe.head(head))
    print("######### Tail #########")
    print(dataframe.tail(head))
    print("######### NA #########")
    print(dataframe.isnull().sum())
    print("######### Quantiles #########")
    print(dataframe.quantile([0,0.05,0.25,0.5,0.99,1]).T)
check_df(df_test) # empty observation -> 0
check_df(df_control) # empty observation-> 0


df_test["Purchase"].mean() # 582.1060966484675
df_control["Purchase"].mean() # 550.8940587702316

# When the average of all values is considered, the test group is larger.
# But is it accidental or due to changes?
# The hypothesis test is applied to question this.


#############################
# A/B Test Hypotheses
#############################

"""
H0: M1 = M2
H0: There is no statistically significant difference between Maximum bidding and Average bidding.

H1: M1 != M2
H1: There is statistically significant difference between Maximum bidding and Average bidding.
"""


####################################
# Hypothesis Testing
####################################

############################
# 1. Assumption Control
############################

# - Normality Assumption
# - Variance Homogeneity


#####################
# Normality Assumption
#####################

# H0: Assumption of normal distribution is provided.
# H1: ... not provided.

# p-value < 0.05 -> H0 rejected
# p-value > 0.05 -> HO not rejected

test_stat, pvalue = shapiro(df_test.loc[:,"Purchase"])
print('Test Stat = %.4f, p-value = %.4f ' % (test_stat, pvalue))
# Result => Test Stat = 0.9589, p-value = 0.1541
# H0 cannot be REJECTED because the p-value is greater than 0.05.

test_stat, pvalue = shapiro(df_control.loc[:,"Purchase"])
print('Test Stat = %.4f, p-value = %.4f ' % (test_stat, pvalue))
# Result => Test Stat = 0.9773, p-value = 0.5891
# H0 cannot be REJECTED because the p-value is greater than 0.05.
# Assumption of normality is provided.

"""
parametric
- It means that the median and the mean are close to each other.

Non-parametric
- It means that the distribution is skewed.
"""


#####################
# Variance Homogeneity
#####################

"""
H0: Assumption of variance homogeneity is provided.
H1: ... not provided.
"""

test_stat, pvalue = levene(df_control["Purchase"],df_test["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Result => Test Stat = 2.6393, p-value = 0.1083
# H0 cannot be REJECTED because the p-value is greater than 0.05.
# Variance homogeneity is provided.

# Parametric test (t-test for two samples) is applied since both assumptions cannot be rejected.


#####################
# Two-Sample T-Test
#####################

test_stat, pvalue = ttest_ind(df_control["Purchase"], df_test["Purchase"], equal_var=True)
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Result => Test Stat = -0.9416, p-value = 0.3493
# H0 cannot be REJECTED because the p-value is greater than 0.05.
# There is no statistical difference between them.
