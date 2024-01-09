import os
from sklearn.linear_model import SGDRegressor
import statsmodels.api as sm
import bio
import numpy as np
import pandas as pd
import glob
import os
import time
import scipy
import multiprocessing
from sklearn.model_selection import cross_val_score, KFold
import pickle
from tqdm import tqdm
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error

"""
Input : ChIPseq-derived trainsets (as a PBM format - signal intensity files)
"""
def print_full(x): # print binary-based sequences for k-mer features
    y = pd.DataFrame(x)
    y.index = [bio.itoseq(x) for x in y.index]
    y["revcomp"] = [bio.revcompstr(x) for x in y.index]
    return y
def variable_seq(row):
    row = row[:36]
    return row
def minmax(score):  # Min-max normalization
    if score.max() == 0:
        return score
    else:
        diff_range =  score.max() - score.min()
        minmax_norm = (score - score.min()) / diff_range
    return minmax_norm
def log2trans(score):  # binary logarithmic scaling
    log_scores = np.log2(score)
    return log_scores
def log2trans_pbm(score):  # binary logarithmic scaling
    minscr = score[score.idxmin()]
    log_scores = np.log2(score - minscr + 1000) #
    return log_scores
def read_chip(pbm_format,norm_method,kmer=6): # reading PBM input and apply transformation to scores and binary seq
    global nonrev_list
    nonrev_list = bio.gen_nonreversed_kmer(kmer) # 2080 features (6-mer DNA)
    pbm_format.columns = ['score','sequence'] #score,sequence ..
    score = norm_method(pbm_format['score']) # log transformation for fluorescent signals
    seqbin = [bio.seqtoi(x) for x in pbm_format['sequence']] #  PBM içindeki her bir sekansı binary gösterimine çevirir
    oligfreq = bio.nonr_olig_freq(seqbin,kmer,nonrev_list) # feature vs sekans içeren count table oluşturur
    return pd.concat([score,oligfreq],axis=1)
def get_cov_params(model,X,y):
    # Calculate sigma^2: σ^2=(Y−Xβ^)T(Y−Xβ^)/(n−p)
    y_hat = model.predict(X) # (predicted values)
    residuals = y - y_hat # lm_pbm.resid (true values - predicted values)
    e = np.array(residuals).reshape(residuals.shape[0], 1) # (Y−Xβ^)
    e_T = np.transpose(e) # (Y−Xβ^)'
    df_res = len(X) - len(nonrev_list) # degree of freedom # n - p # lm_pbm.df_resid
    SSE = np.dot(e_T, e) # the sum of squared errors / the sum of squared residuals : scalar value
    residual_variance =  SSE / (df_res) # lm_pbm.mse_resid (s^2) - MSE estimates sigma^2
    # The covariance matrix of the parameter estimates :
    #  2080*34589 X 34589*2080 --> 2080 * 2080
    if sum(X.apply(np.sum,axis=0) == 0) == 0: # check sum(k-mer feature) = 0 or not?
        scaled_cov_matrix = np.linalg.inv(np.dot(X.T, X)) * residual_variance
    else:
        scaled_cov_matrix = np.linalg.pinv(np.dot(X.T, X)) * residual_variance
        print("\nSingularity problem! Pseudo-inverse of a matrix is used!")
    return scaled_cov_matrix
def apply_ols_metrics_CV(df,TF_name):
    X = df.drop('score',axis=1).apply(minmax,axis=0)
    y = df["score"]
    KF = KFold(n_splits=10, shuffle=True, random_state=25)
    ols_scores = []
    start_time = time.perf_counter()
    for train_index, test_index in KF.split(X):
        X_train, X_test = X.values[train_index], X.values[test_index]
        y_train, y_test = y.values[train_index], y.values[test_index]
        ols = sm.OLS(y_train, X_train).fit()
        y_pred = ols.predict(X_test)
        score = r2_score(y_test, y_pred)
        ols_scores.append(score)
    end_time = time.perf_counter()
    ols_scores = np.array(ols_scores)
    r2 = ols_scores.mean()
    r2_std = ols_scores.std()
    elapsed_time = end_time - start_time
    n_peaks = len(X)
    return [TF_name,n_peaks,r2, r2_std, elapsed_time]
def apply_sgd_metrics_CV(df,TF_name):
    X = df.drop('score', axis=1).apply(minmax, axis=0)
    y = df["score"]
    KF = KFold(n_splits=10, shuffle=True, random_state=25)
    sgd = SGDRegressor(alpha=0.0001, max_iter=1000, tol=1e-3, penalty=None, eta0=0.1, random_state=25)
    start_time = time.perf_counter()
    sgd_scores = cross_val_score(sgd, X, y, cv=KF)
    end_time = time.perf_counter()
    r2 = sgd_scores.mean()
    r2_std = sgd_scores.std()
    elapsed_time = end_time - start_time
    n_peaks = len(X)
    return [TF_name,n_peaks,r2,r2_std,elapsed_time]
def apply_sgd_metrics(df,TF_name):
    X = df.drop('score', axis=1).apply(minmax, axis=0)
    y = df["score"]
    sgd = SGDRegressor(alpha=0.0001, max_iter=1000, tol=1e-3, penalty=None, eta0=0.1, random_state=25)
    start_time = time.perf_counter()
    sgd.fit(X, y)
    end_time = time.perf_counter()
    print_motif = pd.DataFrame({"Weights": sgd.coef_}, index=nonrev_list)  # for array-like output of OLS result
    print_motif = print_full(print_motif["Weights"].sort_values(ascending=False))
    y_pred = sgd.predict(X)
    elapsed_time = end_time - start_time
    mse = mean_squared_error(y, y_pred)
    r = scipy.stats.pearsonr(y, y_pred)[0]
    r2 = r2_score(y, y_pred)
    # Root Mean Squared Error (RMSE)
    rmse = mean_squared_error(y, y_pred, squared=False)
    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(y, y_pred)
    # Median Absolute Error(MEDAE)
    medae = median_absolute_error(y, y_pred)
    n_peaks = len(X)
    # TF_motifs = print_motif.to_csv(f"outputs/PBM_motifs/TF_motifs_SGD/sgd_motif_{TF_name}.csv")
    top_motifs = list(print_motif.index.values[:6])
    top_revmotifs = list(print_motif["revcomp"].values[:6])
    top_weights = print_motif["Weights"].values[:6]
    bottom_motifs = list(print_motif.index.values[-6:])
    bottom_weights = list(print_motif["Weights"].values[-6:])
    return  [TF_name, n_peaks, mse, r, r2, rmse, mae, medae, elapsed_time,
             top_motifs, top_revmotifs,top_weights, bottom_motifs, bottom_weights]  # [1] = coefficients [-1] = covariance
def apply_sgd(df):
    X = df.drop('score',axis=1).apply(minmax,axis=0) # values of features
    y = df["score"] # target values
    sgd = SGDRegressor(loss= "squared_error",alpha=0.0001, max_iter=1000, tol=1e-3, penalty=None, eta0=0.1, random_state=25)
    sgd.fit(X, y)
    cov = get_cov_params(sgd, X, y)
    params = sgd.coef_ # coefficients
    print_motif = pd.DataFrame({"Weights": sgd.coef_}, index=nonrev_list)  # motifs
    print_motif = print_full(print_motif["Weights"].sort_values(ascending=False))
    return sgd,params,print_motif,cov # return[1] = coefficients return[-1] = covariance
def save_params(train_file):
    pbm_format = pd.read_csv(f"{file_path}/{train_file}", sep="\t", header=None)
    ENCODE_ID = train_file.split("_")[1]
    TF_name = train_file.split(".txt")[0].split("_")[-1] # TF extraction
    data_type = train_file.split(".txt")[0].split("_")[0] # Experiment Info
    if type(pbm_format[0][1]) != np.float64: # reverse columns
        pbm_format = pbm_format[[1, 0]]
    # print("\n",len(pbm_format[1][0]),"base pair (bp)")

    if data_type == "PBM":
        df_chip = read_chip(pbm_format, log2trans_pbm)  # count table of trainset
    elif data_type == "ChIPseq":
        df_chip = read_chip(pbm_format, log2trans)  # count table of trainset
    else:
        print("Unrecognized!-----------------!!!!!!!!!-------------------")

    sgd_chip_none = apply_sgd(df_chip)  # run model
    with open(f"outputs/params/p_{train_file}.pkl", "wb") as file:
        pickle.dump(sgd_chip_none, file)
    print(f"outputs/{ENCODE_ID}_{TF_name} is trained and saved!")
    return df_chip
def perf_test(train_file,method="sgd"):
    TF_name = train_file.split(".txt")[0].split("_")[-1] # TF extraction
    data_type = train_file.split(".txt")[0].split("_")[0] # Experiment Info
    pbm_format = pd.read_csv(f"{file_path}/{train_file}", sep="\t", header=None)
    if type(pbm_format[0][1]) != np.float64: # reverse columns
        pbm_format = pbm_format[[1, 0]]
    # print("\n",len(pbm_format[1][0]),"base pair (bp)")
    if data_type == "PBM":
        df_chip = read_chip(pbm_format, log2trans_pbm)  # count table of trainset
    elif data_type == "ChIPseq":
        df_chip = read_chip(pbm_format, log2trans)  # count table of trainset
    else:
        print("Unrecognized!-----------------!!!!!!!!!-------------------")

    if method == "sgd":
        sgd_chip_none = apply_sgd_metrics_CV(df_chip,TF_name)
        return sgd_chip_none
    else:
        ols = apply_ols(df_chip,TF_name)
        return ols
def mypredict(seq1, seq2, k, params,cov_matrix):
    ref = bio.nonr_olig_freq([bio.seqtoi(seq1)],k,nonrev_list)  # from N
    mut = bio.nonr_olig_freq([bio.seqtoi(seq2)],k,nonrev_list)  # to N
    diff_count = mut - ref # c': count matrix
    print("ref score:",np.dot(ref,params))
    print("mut score:",np.dot(mut,params))
    diff = np.dot(diff_count, params) # c'Bhat --> the difference in binding affinity
    print("\ndiff score aka (wildBeta-mutatedBeta) (c'): ",diff)
    SE = np.sqrt((np.dot(diff_count,cov_matrix) * diff_count).sum(axis=1)) # 2080*2080 X 2080*1 = 2080*1 # a scalar value Standart error
    t =  diff/ SE # t-statistic : c'Bhat / sqrt(c'*covBhat*c)
    p_val = scipy.stats.norm.sf(abs(t))*2 # follows t-distribution
    statdict = {"diff":diff[0], "t":t[0], "p_value":p_val[0] }
    return statdict


file_path = "final_trainsets"
trainsets = os.listdir(file_path)

# To display all TF models' performance metrics including top motifs
def process_metric_file(train_file):
    current_process = multiprocessing.current_process().name
    try:
        result = perf_test(train_file, "sgd")
        print(f"{current_process}: {train_file} is trained!\n")
        return result, train_file, None
    except Exception as e:
        return None, train_file, str(e)
def run_metrics(n_process):
    print("---STARTED---\n")
    pool = multiprocessing.Pool(processes=n_process)  # Create a pool of processes
    a = time.perf_counter()
    results = pool.map(process_file, trainsets)  # Map the function across the trainsets
    # perf_df = pd.DataFrame(columns=["TF_Name", "n_peaks", "mse", "r", "r2", "rmse", "mae", "medae", "elapsed_time",
    #                                 "top_motifs", "top_weights", "bottom_motifs", "bottom_weights"])
    perf_df = pd.DataFrame(columns=["TF_Name", "n_peaks", "r2","r2_std","elapsed_time"])
    # perf_df = pd.DataFrame(columns=["TF_Name","n_peaks","mse", "r", "r2", "rmse", "mae", "medae", "elapsed_time","top_motifs","top_revmotifs","top_weights","bottom_motifs","bottom_weights"])

    error_log = []
    for result, train_file, error in results:
        if error:
            error_log.append((train_file, error))
        elif result is not None:
            # df_columns = ["TF_Name","n_peaks","mse", "r", "r2", "rmse", "mae",
            #               "medae", "elapsed_time","top_motifs","top_revmotifs","top_weights","bottom_motifs","bottom_weights"]
            df_columns = ["TF_Name", "n_peaks", "r2","r2_std","elapsed_time"]
            x = pd.DataFrame([result], index=[train_file],
                             columns=df_columns)
            perf_df = pd.concat([perf_df, x])
    print("FINAL:\n")
    print(f"All files is trained!")
    if error_log:
        print("Error log is saved! ")
        with open("outputs/error_log.txt", "w") as file:
            file.write("Errors occurred in the following files:\n")
            for error in error_log:
                file.write(f"File: {error[0]}, Error: {error[1]}\n")
    b = time.perf_counter()
    perf_df.to_csv("outputs/run.csv")
    time_elapsed = b - a
    return perf_df, time_elapsed



# To store parameter estimates and covariance matrix of features trained by SGD
def process_save_file(train_file):
    current_process = multiprocessing.current_process().name
    try:
        result= save_params(train_file)
        print(f"{current_process}: {train_file} is trained!\n")
        return result,train_file, None
    except Exception as e:
        return None, train_file, str(e)
def run_save(n_process):
    print("---STARTED---\n")
    pool = multiprocessing.Pool(processes=n_process)  # Create a pool of processes
    a = time.perf_counter()
    # results = pool.map(process_save_file, trainsets)  # Map the function across the trainsets
    results = []
    # Use tqdm to track the progress of pool.map
    with tqdm(total=len(trainsets), desc='Progress') as pbar:
        for result in pool.imap_unordered(process_save_file, trainsets):
            results.append(result)
            pbar.update()
    error_log = []
    for result,train_file, error in results:
        if error:
            error_log.append((train_file, error))
    if error_log:
        print("Error log is saved! ")
        with open("outputs/error_log.txt", "w") as file:
            file.write("Errors occurred in the following files:\n")
            for error in error_log:
                file.write(f"File: {error[0]}, Error: {error[1]}\n")
    b = time.perf_counter()
    print("FINAL:\n")
    print(f"All files is trained!")
    time_elapsed = b - a
    return time_elapsed


if __name__ == "__main__":
    time_elapsed = run_save(14)
    print("\n")
    print(time_elapsed/60,"min")
    print("\n ---ENDED---")




