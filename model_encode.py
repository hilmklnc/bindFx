import os
from sklearn.linear_model import SGDRegressor
import statsmodels.api as sm
import bio
import numpy as np
import pandas as pd
import glob
import os
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
from sklearn.model_selection import train_test_split
import scipy
import multiprocessing
from sklearn.model_selection import cross_val_score, KFold

# Input : ChIPseq-derived trainset (as a PBM format)
def print_full(x): # print binary-based sequences
    y = pd.DataFrame(x)
    y.index = [bio.itoseq(x) for x in y.index]
    y["revcomp"] = [bio.revcompstr(x) for x in y.index]
    return y
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
def apply_ols(df,TF_name,scaler=minmax):
    X = df.drop('score',axis=1).apply(scaler,axis=0)
    y = df["score"]
    # X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=25)
    kf = KFold(n_splits=10, shuffle=True, random_state=25)
    ols_scores = []
    start_time = time.perf_counter()
    for train_index, test_index in kf.split(X):
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
def apply_sgd_metrics(df, scaler,TF_name):
    X = df.drop('score', axis=1).apply(scaler, axis=0)
    y = df["score"]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4321)
    kf = KFold(n_splits=10, shuffle=True, random_state=25)
    sgd = SGDRegressor(alpha=0.0001, max_iter=1000, tol=1e-3, penalty=None, eta0=0.1,random_state=25)
    start_time = time.perf_counter()
    sgd_scores = cross_val_score(sgd, X, y, cv=kf)
    end_time = time.perf_counter()
    r2 = sgd_scores.mean()
    r2_std = sgd_scores.std()
    # sgd.fit(X, y)
    # print_motif = pd.DataFrame({"Weights": sgd.coef_}, index=nonrev_list)  # for array-like output of OLS result
    # print_motif = print_full(print_motif["Weights"].sort_values(ascending=False))
    # y_pred = sgd.predict(X)
    elapsed_time = end_time - start_time
    # mse = mean_squared_error(y, y_pred)
    # r = scipy.stats.pearsonr(y, y_pred)[0]
    # r2 = r2_score(y_test, y_pred)
    # Root Mean Squared Error (RMSE)
    # rmse = mean_squared_error(y, y_pred, squared=False)
    # Mean Absolute Error (MAE)
    # mae = mean_absolute_error(y, y_pred)
    # Median Absolute Error(MEDAE)
    # medae = median_absolute_error(y, y_pred)
    n_peaks = len(X)
    # TF_motifs = print_motif.to_csv(f"outputs/PBM_motifs/TF_motifs_SGD/sgd_motif_{TF_name}.csv")
    # top_motifs = list(print_motif.index.values[:6])
    # top_weights = print_motif["Weights"].values[:6]
    # bottom_motifs = list(print_motif.index.values[-6:])
    # bottom_weights = list(print_motif["Weights"].values[-6:])
    # [TF_name, n_peaks, mse, r, r2, rmse, mae, medae, elapsed_time, top_motifs, top_weights, bottom_motifs,
    #  bottom_weights]  # [1] = coefficients [-1] = covariance
    return [TF_name,n_peaks,r2,r2_std,elapsed_time]
def perf_test(train_file,method="sgd"):
    # TF_name = train_file.split("\\")[1].upper().split("_")[-1].split(".")[0] qbic
    TF_name = train_file.split(".txt")[0].split("_")[-1] # cisbp
    # TF_name = train_file.split("_")[0].upper() # uniprobe
    pbm_format = pd.read_csv(f"{file_path}/{train_file}", sep="\t", header=None)
    # pbm_format[1] = pbm_format[1].apply(variable_seq)
    if type(pbm_format[0][1]) != np.float64:
        pbm_format = pbm_format[[1, 0]]
    # print("\n",len(pbm_format[1][0]),"base pair (bp)")
    df_chip = read_chip(pbm_format, log2trans)  # count table of trainset
    if method == "sgd":
        sgd_chip_none = apply_sgd_metrics(df_chip, minmax,TF_name)
        return sgd_chip_none
    else:
        ols = apply_ols(df_chip,TF_name)
        return ols
def read_chip(pbm_format,norm_method,kmer=6): # reading PBM input and apply transformation to scores and binary seq
    global nonrev_list
    nonrev_list = bio.gen_nonreversed_kmer(kmer) # 2080 features (6-mer DNA)
    pbm_format.columns = ['score','sequence'] #score,sequence ..
    score = norm_method(pbm_format['score']) # log transformation for fluorescent signals
    seqbin = [bio.seqtoi(x) for x in pbm_format['sequence']] #  PBM içindeki her bir sekansı binary gösterimine çevirir
    oligfreq = bio.nonr_olig_freq(seqbin,kmer,nonrev_list) # feature vs sekans içeren count table oluşturur
    return pd.concat([score,oligfreq],axis=1)
def variable_seq(row):
    row = row[:36]
    return row
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
    scaled_cov_matrix = np.linalg.inv(np.dot(X.T, X)) * residual_variance
    return pd.DataFrame(scaled_cov_matrix)
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


# TF_name = train_file.split("_")[0].upper() # uniprobe
# pbm_format[1] = pbm_format[1].apply(variable_seq)

# trainsets = glob.glob("pbm_trainset_v1/*.txt")


# os.makedirs("outputs/PBM_motifs"
# os.makedirs("outputs/PBM_motifs/TF_motifs_OLS")
# os.makedirs("outputs/PBM_motifs/TF_motifs_SGD")
file_path= "outputs/encode_trainsets"
trainsets = os.listdir(file_path)
# trainsets = glob.glob("TFs_trainset/*.txt")
# def run_all():
#     global perf_df
#     perf_df = pd.DataFrame(columns=["TF_Name","n_peaks","mse", "r", "r2", "rmse", "mae", "medae", "elapsed_time","top_motifs","top_weights","bottom_motifs","bottom_weights"])
#     for i,train_file in enumerate(trainsets):
#         # name = train_file.split("\\")[1]
#         x = perf_test(train_file,"sgd")
#         new = pd.DataFrame([x], index=[train_file], columns=["TF_Name","n_peaks","mse", "r","r2", "rmse", "mae", "medae", "elapsed_time","top_motifs","top_weights","bottom_motifs","bottom_weights"])
#         perf_df = pd.concat([perf_df, new])
#         print(f"{i+1}th: {train_file} is trained!")
#         print("\n")
#     perf_df.to_csv("outputs/encode_all_results.csv")
#     return  perf_df
#
# result = run_all()

def process_file(train_file):
    current_process = multiprocessing.current_process().name
    try:
        result = perf_test(train_file, "ols")
        print(f"{current_process}: {train_file} is trained!\n")
        return result, train_file, None
    except Exception as e:
        return None, train_file, str(e)
def run_all(n_process):
    print("---STARTED---\n")
    pool = multiprocessing.Pool(processes=n_process)  # Create a pool of processes
    a = time.perf_counter()
    results = pool.map(process_file, trainsets)  # Map the function across the trainsets
    # perf_df = pd.DataFrame(columns=["TF_Name", "n_peaks", "mse", "r", "r2", "rmse", "mae", "medae", "elapsed_time",
    #                                 "top_motifs", "top_weights", "bottom_motifs", "bottom_weights"])
    perf_df = pd.DataFrame(columns=["TF_Name", "n_peaks", "r2","r2_std","elapsed_time"])
    error_log = []
    for result, train_file, error in results:
        if error:
            error_log.append((train_file, error))
        elif result is not None:
            # x = pd.DataFrame([result], index=[train_file],
            #                  columns=["TF_Name", "n_peaks", "mse", "r", "r2", "rmse", "mae",
            #                           "medae", "elapsed_time", "top_motifs", "top_weights",
            #                           "bottom_motifs", "bottom_weights"])
            x = pd.DataFrame([result], index=[train_file],
                             columns=["TF_Name", "n_peaks", "r2","r2_std","elapsed_time"])
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
    perf_df.to_csv("outputs/encode_all_results_OLS20.csv")
    time_elapsed = b - a
    return perf_df, time_elapsed

if __name__ == "__main__":
    result,time_elapsed = run_all(n_process)
    print("\n")
    print("The total trained files:", len(result))
    print(result.head())
    print(time_elapsed/60,"min")
    print("\n ---ENDED---")

#
# def run_all():
#     global perf_df
#     a = time.time()
#     perf_df = pd.DataFrame(columns=["TF_Name", "n_peaks", "mse", "r", "r2", "rmse", "mae", "medae", "elapsed_time",
#                                     "top_motifs", "top_weights", "bottom_motifs", "bottom_weights"])
#     error_log = []  # Store information about errors encountered
#
#     for i, train_file in enumerate(trainsets):
#         try:
#             x = perf_test(train_file, "sgd")
#             new = pd.DataFrame([x], index=[train_file], columns=["TF_Name", "n_peaks", "mse", "r", "r2", "rmse", "mae",
#                                                                  "medae", "elapsed_time", "top_motifs", "top_weights",
#                                                                  "bottom_motifs", "bottom_weights"])
#             perf_df = pd.concat([perf_df, new])
#             print(f"{i+1}th: {train_file} is trained!")
#             print("\n")
#         except Exception as e:
#             error_log.append((train_file, str(e)))
#             print(f"Error occurred with {train_file}: {e}. Skipping this file.")
#
#     if error_log:
#         with open("outputs/error_log.txt", "w") as file:
#             file.write("Errors occurred in the following files:\n")
#             for error in error_log:
#                 file.write(f"File: {error[0]}, Error: {error[1]}\n")
#     perf_df.to_csv("outputs/encode_all_results.csv")
#     b = time.time()
#     time_elapsed = b - a
#     return perf_df,time_elapsed
#
#
#
#
#
#
#
#

# Upload Sample File
# Exlusively
pbm_format = pd.read_csv(f"Homo_sapiens_NA_Unpublished_GATA1.txt", sep="\t", header=None)
# pbm_format[1] = pbm_format[1].apply(variable_seq)
df = read_chip(pbm_format, log2trans)  # count table of trainset
X = df.drop('score',axis=1).apply(minmax,axis=0) # values of features
y = df["score"] # target values
print(df)
#
len(df)
len(df.columns)

bio.itoseq(4099)
bio.seqtoi("GATAAG")
    # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
# #
# # # Run Model
sgd = SGDRegressor(loss="squared_error",alpha=0.0001, max_iter=1000, tol=1e-3, penalty=None, eta0=0.1,random_state=10)
# kf = KFold(n_splits=10,shuffle=True,random_state=25)
# sgd_scores = cross_val_score(sgd, X, y, cv=kf,verbose=1,n_jobs=-1)
# sgd_scores
# sgd_scores.mean()
# sgd_scores.std()
#

sgd.fit(X,y)
print_motif = pd.DataFrame({"Weights": sgd.coef_}, index=nonrev_list)  # for array-like output of OLS result
print_motif = print_full(print_motif["Weights"].sort_values(ascending=False))
cov_matrix = get_cov_params(sgd,X, y)
seq1 = "GCAATTTCAGTCTACAGCATGTGCATGCTTATCAGTGCATTCTAAATATTTCTATGTGAG".upper()
seq2 = "GCAATTTCAGTCTACAGCATGTGCATGCTTCTCAGTGCATTCTAAATATTTCTATGTGAG".upper() #  8.279763929029813e-20
mypredict(seq1, seq2, 6,sgd.coef_,cov_matrix)

# # Independent Test dataset
# pbm_format_test = pd.read_csv("Homo_sapiens_M00558_1.94d_Barrera2016_EGR2_E412K_R1.txt", sep="\t", header=None)
# df_test = read_chip(pbm_format_test, log2trans)
# X_test = df_test.drop('score',axis=1).apply(minmax,axis=0) # values of features
# y_test = df_test["score"]
# sgd_y = sgd.predict(X)
# r2_score(y,sgd_y)
#
# sgd_y = sgd.predict(X_test)
# scipy.stats.pearsonr(y_test, sgd_y)[0]
# r2_score(y_test,sgd_y)
# mean_squared_error(y_test,sgd_y)
#
# ols_pbm = sm.OLS(y_train,X_train).fit()
# param_pbm = print_full(ols_pbm.params.sort_values(ascending=False))
#
# cov_matrix = get_cov_params(ols_pbm,X,y)
# params = ols_pbm.params
# a = ols_pbm.predict(X_test)
# mean_squared_error(y_test,a)
# r2_score(y_test,a)
# scipy.stats.pearsonr(y_test, a)[0]
#

# Select the highest r2 performance between each others
perf_df = pd.read_csv("outputs/encode_unique_maxr2_CV.csv",index_col="Unnamed: 0")
len(perf_df["TF_Name"])
len(perf_df["TF_Name"].unique())
TF_names = perf_df["TF_Name"].unique()

targets = []
for tf in TF_names:
    ind = perf_df[perf_df["TF_Name"] == tf]["r2"].idxmax()
    targets.append(ind)
len(targets) # 254 unique TF datasets
perf_df.loc[targets].to_csv("outputs/encode_unique_maxr2_CV.csv")

pd.Series(targets).to_csv("outputs/target_filenames.csv")

list1 = targets
list2 = target_1
difference_set = set(list2) ^ set(list1)
common_elements = set(list2) & set(list1)
len(difference_set)
len(common_elements)
targets == target_1
target_1 = targets
len(target)
sum(perf_df["n_peaks"] > 2080)
sum(perf_df["r2"] >= 0.1)
sum(perf_df[perf_df["r2"] > 0.1]["n_peaks"]  > 2080)

chip_filtered = perf_df[perf_df["r2"] > 0.1]["TF_Name"].unique()

pbm_tfs = pd.read_csv("outputs/prev/pbm_all_results_maxr2.csv")
pbm_uniq = pbm_tfs["TF_Name"].unique()
chip_uniq = perf_df["TF_Name"].unique()

common_elements = set(pbm_uniq) & set(chip_uniq)
difference_set = set(pbm_uniq) ^ set(chip_uniq)
len(common_elements)
len(difference_set)
