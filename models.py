import pandas as pd
import numpy as np
import bio
import scipy.stats
from time import time
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import statsmodels.api as sm

def read_pbm(filename,kmer,nonrev_list,norm_method,gappos=0,gapsize=0): # reading PBM input and apply transformation for scores and binary seq
    tbl = pd.read_csv(filename,names=['score','sequence'],delim_whitespace=True) #score,sequence ..
    score = norm_method(tbl['score']) # log transformation for fluorescent signals
    seqbin = [bio.seqtoi(x) for x in tbl['sequence']] #  PBM içindeki her bir sekansı binary gösterimine çevirir
    oligfreq = bio.nonr_olig_freq(seqbin,kmer,nonrev_list,gappos=gappos,gapsize=gapsize) # feature vs sekans içeren count table oluşturur
    return pd.concat([score,oligfreq],axis=1)
def robustscaler(score): #Perform robust standardization, scale according to IQR
    scaler = RobustScaler()
    data= score.to_numpy().reshape(-1, 1)
    scaler.fit(data)
    return pd.DataFrame(scaler.transform(data)).squeeze().rename("score")
# X-Xmedian / X75 - X25 (IQR)

def standartscaler(score): # z-score transformation
    return (score -score.mean()) / score.std()

def minmax(score):  # Min-max normalization
    diff_range =  score.max() - score.min()
    minmax_norm = (score - score.min()) / diff_range
    return minmax_norm

def powertransform(score): # Apply a power transform featurewise to make data more Gaussian-like.
    scaler = PowerTransformer() # sd = 1, mean = 0 like standartscaler
    data = score.to_numpy().reshape(-1, 1)
    scaler.fit(data)
    return pd.DataFrame(scaler.transform(data)).squeeze().rename("score")
def logshift(score):  # logarithmic scaling
    minscr = score[score.idxmin()] # pbm datasetindeki min değeri
    # if score.median() == 0: # when median of raw scores equals 0
    shift = 1000 # adding log(0 + 1)
    # else:
    #     shift = score.median()
    log_scores = np.log(score - minscr + shift) # log scaling   Log (SCORE - Minimum score + shift miktarı)
    return log_scores # shift değeri min değerini sıfırdan kurtarmak için eklenilen pay

def logtrans(score):  # logarithmic scaling
    log_scores = np.log2(score)
    return log_scores
def logtrans2(score):  # logarithmic scaling
    log_scores = np.log2(score+1)
    return log_scores


def print_weights_by_order(weights, features):  # prints features by order of weights
    weights = weights.flatten()
    series = pd.Series(data=weights, index=features)
    print(series.sort_values(ascending=False))
def no_scaling(score):
    return score
def print_full(x):
    y = pd.DataFrame(x)
    y.index = [bio.itoseq(x) for x in y.index]
    y["revcomp"] = [bio.revcompstr(x) for x in y.index]
    return y
# Initiate
k = 6
nonrev_list = bio.gen_nonreversed_kmer(k) # 2080 features (6-mer DNA)

# Dataframes:
df_pbm = read_pbm("/Users/husey/qbic/PBM_inputs/Homo_sapiens_NA_Unpublished_GATA1.txt", k, nonrev_list, logshift)  # 34589 samples * 2080 features
df_chip_ENCFF853VZF = read_pbm("/Users/husey/qbic/GATA1/GATA1chip_pbmformat/ChIPseq_ENCFF853VZF_GATA1.txt", k, nonrev_list, no_scaling)  # 38463 samples * 2080 features
df_chip_ENCFF853VZF_log = read_pbm("/Users/husey/qbic/GATA1/GATA1chip_pbmformat/ChIPseq_ENCFF853VZF_GATA1.txt", k, nonrev_list, logtrans)  # 38463 samples * 2080 features
df_chip_negVZF = read_pbm("/Users/husey/qbic/trainset_VZF_file2", k, nonrev_list, no_scaling)  # 76926 samples * 2080 features
df_chip_ENCFF657CTC = read_pbm("/Users/husey/qbic/GATA1/GATA1chip_pbmformat/ChIPseq_ENCFF657CTC_GATA1.txt", k, nonrev_list, no_scaling)  # 14676 samples * 2080 features
df_chip_ENCFF853VZF_ENCFF657CTC = read_pbm("/Users/husey/qbic/GATA1/GATA1chip_pbmformat/VZF_CTC_combined.txt", k, nonrev_list, no_scaling)  # 53139 samples * 2080 features
df_chip_neg_VZF_CTC = read_pbm("/Users/husey/qbic/GATA1/GATA1chip_pbmformat/negatived_VZF_CTC", k, nonrev_list, no_scaling)  # 115484 samples * 2080 features

# Dataframes 2:
df_chip_negVZF_2 = read_pbm("trainset_negatived_vzf", k, nonrev_list, logtrans)  # 76926 samples * 2080 features
df_chip_ENCFF657CTC_log = read_pbm("/Users/husey/qbic/GATA1/GATA1chip_pbmformat/ChIPseq_ENCFF657CTC_GATA1.txt", k, nonrev_list, logtrans)  # 14676 samples * 2080 features
df_chip_ENCFF853VZF_log2 = read_pbm("/Users/husey/qbic/GATA1/GATA1chip_pbmformat/ChIPseq_ENCFF853VZF_GATA1.txt", k, nonrev_list, logtrans2)  # 38463 samples * 2080 features


# Scalers: # lets check other feature-scalings
X = df_chip_ENCFF853VZF_log.drop('score',axis=1).apply(minmax).values #minmax
# X = df_chip.drop('score',axis=1).apply(robustscaler, axis=0) # robust scaler
# X = df_chip.drop('score',axis=1).apply(standartscaler,axis=0)
# X = df_chip.drop('score',axis=1).apply(powertransform, axis=0)
# X = df_chip.drop('score',axis=1).apply(maxscaler, axis=0)
y = df_chip_ENCFF853VZF_log['score'].values

X = df_pbm.drop('score',axis=1).apply(minmax) # count matrix - explanatory variables
print(X.shape) # (34589, 2080) - count matrix
y = df_pbm['score']
print(y.shape) # (34589, ) - # target values
# Fit OLS model X,y
lm_pbm = sm.OLS(y,X).fit()
print_full(lm_pbm.params.sort_values(ascending=False))
X = df_chip_VZF.drop('score',axis=1)
X = df_pbm.drop("score",axis=1).apply(minmax)
y = df_chip_negVZF['score']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
lm1 = sm.OLS(y,X).fit()
y_pred = lm_pbm.predict(X)

# Check for different penalties 
sgd_reg = SGDRegressor(alpha=0.0001,max_iter=1000, tol=1e-3, penalty=None, eta0=0.1, random_state=25)
sgd_reg.fit(X,y)
# sgd_reg.get_params()
params = pd.DataFrame({"Weights":sgd_reg.coef_},index=nonrev_list) #  for array-like output of OLS result
params = print_full(params["Weights"].sort_values(ascending=False))
y_pred = sgd_reg.predict(X)
# Calculate the mean squared error of the predictions
mse = mean_squared_error(y, y_pred)
# print("Mean squared error:", mse)

def apply_sgd(df,scaler,lss="squared_error",regularizer=None):
    X = df.drop('score',axis=1).apply(scaler,axis=0)
    y = df["score"]
    sgd_reg = SGDRegressor(loss= lss,alpha=0.0001, max_iter=1000, tol=1e-3, penalty=regularizer, eta0=0.1, random_state=25)
    start_time = time()
    sgd_reg.fit(X, y)
    end_time = time()
    elapsed_time = end_time - start_time
    params = pd.DataFrame({"Weights": sgd_reg.coef_}, index=nonrev_list)  # for array-like output of OLS result
    params = print_full(params["Weights"].sort_values(ascending=False))
    y_pred = sgd_reg.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y,y_pred)
    return params,mse, r2,elapsed_time

def apply_ols(df,scaler):
    X = df.drop('score',axis=1).apply(scaler,axis=0)
    y = df["score"]
    start_time = time()
    lm = sm.OLS(y, X).fit()
    end_time = time()
    elapsed_time = end_time - start_time
    params = print_full(lm.params.sort_values(ascending=False))
    y_pred = lm.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y,y_pred)
    return params,mse, r2,elapsed_time


# PBM file - no scaling
ols_pbm = apply_ols(df_pbm,no_scaling)
sgd_pbm = apply_sgd(df_pbm, no_scaling)
# PBM file - minmax
ols_pbm = apply_ols(df_pbm, no_scaling)
sgd_pbm_none = apply_sgd(df_pbm,no_scaling,"squared_error",None)
sgd_pbm_l1 = apply_sgd(df_pbm,no_scaling,"squared_error","l1")
sgd_pbm_l2 = apply_sgd(df_pbm,no_scaling,"squared_error","l2")
sgd_pbm_els = apply_sgd(df_pbm,no_scaling,"squared_error","elasticnet")


# None,L1,L2,elasticnet
# Single VZF - minmax
ols_vzf = apply_ols(df_chip_ENCFF853VZF, robustscaler)
sgd_vzf_none = apply_sgd(df_chip_ENCFF853VZF,robustscaler,"squared_error",None)
sgd_vzf_l1 = apply_sgd(df_chip_ENCFF853VZF,robustscaler,"squared_error","l1")
sgd_vzf_l2 = apply_sgd(df_chip_ENCFF853VZF,robustscaler,"squared_error","l2")
sgd_vzf_els = apply_sgd(df_chip_ENCFF853VZF,robustscaler,"squared_error","elasticnet")

# Single VZF - Power-transform
ols_vzf_pt = apply_ols(df_chip_ENCFF853VZF,powertransform)
sgd_vzf_pt = apply_sgd(df_chip_ENCFF853VZF, powertransform)
# Single VZF - Robust scaler
ols_vzf_rs = apply_ols(df_chip_ENCFF853VZF,robustscaler)
sgd_vzf_rs = apply_sgd(df_chip_ENCFF853VZF, robustscaler)
# Single VZF - Standart scaler
# ols_vzf_ss = apply_ols(df_chip_ENCFF853VZF,standartscaler)
# sgd_vzf_ss = apply_sgd(df_chip_ENCFF853VZF, standartscaler)

# Negatived VZF - minmax
ols_vzf = apply_ols(df_chip_negVZF, minmax)
sgd_vzf = apply_sgd(df_chip_negVZF, minmax)
# Single VZF - Power-transform
ols_vzf_pt = apply_ols(df_chip_negVZF,powertransform)
sgd_vzf_pt = apply_sgd(df_chip_negVZF, powertransform)
# Single VZF - Robust scaler
ols_vzf_rs = apply_ols(df_chip_negVZF,robustscaler)
sgd_vzf_rs = apply_sgd(df_chip_negVZF, robustscaler)

# Single CTC - minmax
ols_vzf = apply_ols(df_chip_ENCFF657CTC, minmax)
sgd_vzf = apply_sgd(df_chip_ENCFF657CTC, minmax)
# Single VZF - Power-transform
ols_vzf_pt = apply_ols(df_chip_ENCFF657CTC,powertransform)
sgd_vzf_pt = apply_sgd(df_chip_ENCFF657CTC, powertransform)
# Single VZF - Robust scaler
ols_vzf_rs = apply_ols(df_chip_ENCFF657CTC,robustscaler)
sgd_vzf_rs = apply_sgd(df_chip_ENCFF657CTC, robustscaler)


# VZF + CTC files - minmax
ols_vzf = apply_ols(df_chip_ENCFF853VZF_ENCFF657CTC, minmax)
sgd_vzf = apply_sgd(df_chip_ENCFF853VZF_ENCFF657CTC, minmax)
# Single VZF - Power-transform
ols_vzf_pt = apply_ols(df_chip_ENCFF853VZF_ENCFF657CTC,powertransform)
sgd_vzf_pt = apply_sgd(df_chip_ENCFF853VZF_ENCFF657CTC, powertransform)
# Single VZF - Robust scaler
ols_vzf_rs = apply_ols(df_chip_ENCFF853VZF_ENCFF657CTC,robustscaler)
sgd_vzf_rs = apply_sgd(df_chip_ENCFF853VZF_ENCFF657CTC, robustscaler)



# df_chip_neg_VZF_CTC -minmax
ols_vzf = apply_ols(df_chip_neg_VZF_CTC, minmax)
sgd_vzf = apply_sgd(df_chip_neg_VZF_CTC, minmax)
# Single VZF - Power-transform
ols_vzf_pt = apply_ols(df_chip_neg_VZF_CTC,powertransform)
sgd_vzf_pt = apply_sgd(df_chip_neg_VZF_CTC, powertransform)
# Single VZF - Robust scaler
ols_vzf_rs = apply_ols(df_chip_neg_VZF_CTC,robustscaler)
sgd_vzf_rs = apply_sgd(df_chip_neg_VZF_CTC, robustscaler)


# df_chip_negVZF_2
ols_vzf = apply_ols(df_chip_negVZF_2, minmax)
sgd_vzf = apply_sgd(df_chip_negVZF_2, minmax)
# Single VZF - Power-transform
ols_vzf_pt = apply_ols(df_chip_negVZF_2,powertransform)
sgd_vzf_pt = apply_sgd(df_chip_negVZF_2, powertransform)
# Single VZF - Robust scaler
ols_vzf_rs = apply_ols(df_chip_negVZF_2,robustscaler)
sgd_vzf_rs = apply_sgd(df_chip_negVZF_2, robustscaler)


# Single VZF - minmax  + log2
ols_vzf = apply_ols(df_chip_ENCFF853VZF_log, minmax)
sgd_vzf = apply_sgd(df_chip_ENCFF853VZF_log, minmax)
# Single VZF - Power-transform
ols_vzf_pt = apply_ols(df_chip_ENCFF853VZF_log,powertransform)
sgd_vzf_pt = apply_sgd(df_chip_ENCFF853VZF_log, powertransform)
# Single VZF - Robust scaler
ols_vzf_rs = apply_ols(df_chip_ENCFF853VZF_log,robustscaler)
sgd_vzf_rs = apply_sgd(df_chip_ENCFF853VZF_log, robustscaler)
# Single VZF - Standart scaler
ols_vzf_ss = apply_ols(df_chip_ENCFF853VZF_log,standartscaler)
sgd_vzf_ss = apply_sgd(df_chip_ENCFF853VZF_log, standartscaler)

# ----------------------------------------------------------------------------
# Single VZF - log2 + minmax
ols_vzf = apply_ols(df_chip_ENCFF853VZF_log, minmax)
sgd_vzf_none = apply_sgd(df_chip_ENCFF853VZF_log,minmax,"squared_error",None)
sgd_vzf_l1 = apply_sgd(df_chip_ENCFF853VZF_log,minmax,"squared_error","l1")
sgd_vzf_l2 = apply_sgd(df_chip_ENCFF853VZF_log,minmax,"squared_error","l2")
sgd_vzf_els = apply_sgd(df_chip_ENCFF853VZF_log,minmax,"squared_error","elasticnet")


# Single VZF - log2 + no feature scaling
ols_vzf = apply_ols(df_chip_ENCFF853VZF_log, no_scaling)
sgd_vzf_none = apply_sgd(df_chip_ENCFF853VZF_log,no_scaling,"squared_error",None)
sgd_vzf_l1 = apply_sgd(df_chip_ENCFF853VZF_log,no_scaling,"squared_error","l1")
sgd_vzf_l2 = apply_sgd(df_chip_ENCFF853VZF_log,no_scaling,"squared_error","l2")
sgd_vzf_els = apply_sgd(df_chip_ENCFF853VZF_log,no_scaling,"squared_error","elasticnet")

# running time of your model


# Single VZF - log2 + no feature scaling
ols_vzf = apply_ols(df_chip_ENCFF853VZF_log, robustscaler)
sgd_vzf_none = apply_sgd(df_chip_ENCFF853VZF_log,robustscaler,"squared_error",None)
sgd_vzf_l1 = apply_sgd(df_chip_ENCFF853VZF_log,robustscaler,"squared_error","l1")
sgd_vzf_l2 = apply_sgd(df_chip_ENCFF853VZF_log,robustscaler,"squared_error","l2")
sgd_vzf_els = apply_sgd(df_chip_ENCFF853VZF_log,robustscaler,"squared_error","elasticnet")



# Single CTC - log2 + minmax
ols_vzf = apply_ols(df_chip_ENCFF657CTC_log, minmax)
sgd_vzf_none = apply_sgd(df_chip_ENCFF657CTC_log,minmax,"squared_error",None)
sgd_vzf_l1 = apply_sgd(df_chip_ENCFF657CTC_log,minmax,"squared_error","l1")
sgd_vzf_l2 = apply_sgd(df_chip_ENCFF657CTC_log,minmax,"squared_error","l2")
sgd_vzf_els = apply_sgd(df_chip_ENCFF657CTC_log,minmax,"squared_error","elasticnet")

def apply_bayes(df,scaler,lss="squared_error",regularizer=None):
    X = df.drop('score',axis=1).apply(scaler,axis=0)
    y = df["score"]
    start_time = time()
    lm = linear_model.BayesianRidge()
    lm.fit(X, y)
    end_time = time()
    elapsed_time = end_time - start_time
    params = pd.DataFrame({"Weights": lm.coef_}, index=nonrev_list)  # for array-like output of OLS result
    params = print_full(params["Weights"].sort_values(ascending=False))
    y_pred = lm.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y,y_pred)
    return params,mse, r2,elapsed_time

def apply_sgd(df,scaler,lss="squared_error",regularizer=None):
    X = df.drop('score',axis=1).apply(scaler,axis=0)
    y = df["score"]
    start_time = time()
    sgd_reg = SGDRegressor(loss= lss,alpha=0.00001, max_iter=1000, tol=1e-5, penalty=regularizer, eta0=0.1, learning_rate="adaptive",random_state=333)
    sgd_reg.fit(X, y)
    end_time = time()
    elapsed_time = end_time - start_time
    params = pd.DataFrame({"Weights": sgd_reg.coef_}, index=nonrev_list)  # for array-like output of OLS result
    params = print_full(params["Weights"].sort_values(ascending=False))
    y_pred = sgd_reg.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y,y_pred)
    return params,mse, r2,elapsed_time


bayes_vzf = apply_bayes(df_chip_ENCFF853VZF_log, minmax)


ols_vzf = apply_ols(df_chip_ENCFF853VZF_log, minmax)
sgd_vzf_none = apply_sgd(df_chip_ENCFF853VZF_log2,minmax,"squared_error",None)
sgd_vzf_l1 = apply_sgd(df_chip_ENCFF853VZF_log,minmax,"squared_error","l1")
sgd_vzf_l2 = apply_sgd(df_chip_ENCFF853VZF_log2,minmax,"squared_error","l2")
sgd_vzf_els = apply_sgd(df_chip_ENCFF853VZF_log2,minmax,"squared_error","elasticnet")

loss_grid = ["squared_error", "huber"]
penalty_grid = [None,"l1","l2","elasticnet"]
tolerance_grid = [10e-3,10e-4,10e-5]
learning_grid = ["invscaling","adaptive"]
alpha_grid = [10e-3,10e-4,10e-5]
parameters = {'loss': loss_grid,
              'penalty': penalty_grid,
              'tol':tolerance_grid,
              "learning_rate":learning_grid,
              "alpha":alpha_grid}

gridCV = GridSearchCV(SGDRegressor(),
                      param_grid=parameters,
                      n_jobs=-1)

gridCV.fit(X, y)
best_loss_estim = gridCV.best_params_['loss']
best_penalty = gridCV.best_params_['penalty']
best_tolerance = gridCV.best_params_['tol']
best_learning_rate = gridCV.best_params_['learning_rate']
best_alpha = gridCV.best_params_['alpha']

""""
best_loss_estim
Out[125]: 'squared_error'
best_penalty
Out[126]: 'l1'
best_tolerance
Out[127]: 0.01
best_learning_rate
Out[128]: 'invscaling'
best_alpha
Out[130]: 0.01
"""