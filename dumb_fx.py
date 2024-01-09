#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Fri Dec 23 13:03:56 2022

@author: hilmi

"""

import numpy as np
import pandas as pd
import pyfastx
import scipy.stats
import statsmodels.api as sm
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import RobustScaler

import bio


# SCALER METHODS

def maxscaler(score): # Scale each feature by its maximum absolute value.
    scaler = MaxAbsScaler()
    data = score.to_numpy().reshape(-1, 1)
    scaler.fit(data)
    return pd.DataFrame(scaler.transform(data)).squeeze().rename("score")
# chipseq_train["score"]/chipseq_train["score"].max()
# maxscaler(chipseq_train["score"])

def powertransform(score): # Apply a power transform featurewise to make data more Gaussian-like.
    scaler = PowerTransformer() # sd = 1, mean = 0 like standartscaler
    data = score.to_numpy().reshape(-1, 1)
    scaler.fit(data)
    return pd.DataFrame(scaler.transform(data)).squeeze().rename("score")

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

def logshift(score):  # logarithmic scaling
    minscr = score[score.idxmin()] # pbm datasetindeki min değeri
    # if score.median() == 0: # when median of raw scores equals 0
    shift = 1000 # adding log(0 + 1)
    # else:
    #     shift = score.median()
    log_scores = np.log2(score - minscr + shift) # log scaling   Log (SCORE - Minimum score + shift miktarı)
    return log_scores # shift değeri min değerini sıfırdan kurtarmak için eklenilen pay
def no_scaling(score):
    return score

def read_pbm(filename,kmer,nonrev_list,norm_method,gappos=0,gapsize=0): # reading PBM input and apply transformation for scores and binary seq
    tbl = pd.read_csv(filename,names=['score','sequence'],delim_whitespace=True) #score,sequence ..
    score = norm_method(tbl['score']) # log transformation for fluorescent signals
    seqbin = [bio.seqtoi(x) for x in tbl['sequence']] #  PBM içindeki her bir sekansı binary gösterimine çevirir
    oligfreq = bio.nonr_olig_freq(seqbin,kmer,nonrev_list,gappos=gappos,gapsize=gapsize) # feature vs sekans içeren count table oluşturur
    return pd.concat([score,oligfreq],axis=1)

def read_chipseq(data,kmer,nonrev_list,seqtype,norm_method,gappos=0,gapsize=0): # reading chipseq input and apply transformation for scores and binary seq
    tbl = data #score,sequence ..
    score = norm_method(tbl["score"]) # log transformation for scores
    seqbin = [bio.seqtoi(x) for x in tbl[seqtype]] #  bed location içindeki her bir sekansı binary gösterimine çevirir
    oligfreq = bio.nonr_olig_freq(seqbin,kmer,nonrev_list,gappos=gappos,gapsize=gapsize) # feature vs sekans içeren count table oluşturur
    return pd.concat([score,oligfreq],axis=1)  # pd.concat([score,oligfreq],axis=1) # for normalized

def print_full(x):
    y = pd.DataFrame(x)
    y.index = [bio.itoseq(x) for x in y.index]
    y["revcomp"] = [bio.revcompstr(x) for x in y.index]
    return y

def peak_offset(sequence,peak,offset):
    peak = peak-1
    left_offset = peak-offset
    right_offset = len(sequence) - (peak+offset)
    if (left_offset) < 0:
        start = 0
        end = (peak + offset + abs(left_offset))
    elif (right_offset) < 0:
        start = (peak - (offset + abs(right_offset)))
        end = (peak+offset)
    else:
        start = (left_offset)
        end = peak+offset
    return sequence[start : end]

def mypredict2(seq1, seq2, lm):
    ref = bio.nonr_olig_freq([bio.seqtoi(seq1)],6,nonrev_list, 0, 0)  #*AC
    mut = bio.nonr_olig_freq([bio.seqtoi(seq2)],6,nonrev_list, 0, 0)  # GT
    diff_count = mut - ref # c'
    print("wild type score:", lm.predict(ref)[0])
    print("mutated type score:", lm.predict(mut)[0])
    print("count table of difference(c'):\n",diff_count)
    diff = lm.predict(diff_count) # c'Bhat --> Xi inputs
    sd_diff = (diff_count.transpose() * np.dot(lm.cov_params(),diff_count.transpose())).sum(axis=0).apply(np.sqrt)
    print("standart difference(denominator):\n",sd_diff[0])
    t = diff / sd_diff # t-statistic : c'Bhat / sqrt(c'*covBhat*c)
    p_val = scipy.stats.norm.sf(abs(t))*2 # follows t-distribution
    df = pd.DataFrame({"diff": [diff[0]], "t":[t[0]],"p_value":[p_val[0]]}, columns=["diff","t","p_value"])
    print(df)
    return df

def model_run(df,scale):
    if scale == "minmax":
        X = minmax(df.drop('score', axis=1))
    else:
        X = df.drop('score', axis=1).apply(powertransform, axis=0)
    y = df['score']
    # Fit OLS model X,y
    ols_pbm = sm.OLS(y, X).fit()
    return ols_pbm
def print_result(df,scale,str1,str2):
    model = model_run(df, scale)
    # param = print_full(model.params.sort_values(ascending=False))
    # print(param)
    mypredict(str1, str2, model)


#---------------------------------------------------------------------------------------------------------------#
k = 6 # k-mers
nonrev_list = bio.gen_nonreversed_kmer(k) # 2080 features (6-mer DNA)

# PBM-BASED TRAINING
# Count/frequency matrix
# df_pbm = read_pbm("/Users/husey/qbic/PBM_inputs/Homo_sapiens_NA_Unpublished_GATA1.txt", k, nonrev_list, logshift)  # 34589 samples * 2080 features
#
# # Feature-scaling OFF
# X = df_pbm.drop('score',axis=1) # count matrix - explanatory variables
# print(X.shape) # (34589, 2080) - count matrix
# y = df_pbm['score']
# print(y.shape) # (34589, ) - # target values
# # Fit OLS model X,y
# lm_pbm = sm.OLS(y,X_scaled).fit()
#
# # Feature-scaling ON
# X_scaled = minmax(df_pbm.drop('score',axis=1))
# # Check L- k +1 frequency for each row
# # df_pbm.drop("score", axis=1).apply(sum,axis=1) # 60 - 6 + 1 = 55
# param_pbm = print_full(lm_pbm.params.sort_values(ascending=False))
# param_pbm.columns = ["Weights", "Revcomp"]
# print(param_pbm)

#--------------------------------------------------------------------
# Writing into report
# with open("lm_pbm_summary_GATA1_featurescaledMinMax.txt","w") as f:
#     f.write(lm_pbm.summary().as_text())
#     f.write("\nWeights of Features:\n")
#     f.write(str(param_pbm))
#     f.write("\nMean Squared Error:\n")
#     f.write(str(MSE_pbm))
#
# with open("summary_chipseq_powertransform.txt","w") as f:
#     f.write(lm_chip.summary().as_text())
#     f.write("\nWeights of Features:\n")
#     f.write(str(param_chip[:15]))
#     f.write("\ntail:\n")
#     f.write(str(param_chip[2075:]))
#     f.write("\nMean Squared Error:\n")
#     f.write(str(MSE_chip))

#--------------------------------------------------------------------------------
# CTTATC bölgelerini çıkartıp test et
# DNase + H3K4me1 + H3K4me2 + H3K4me3 + H3K9ac + H3K27ac (K562 cell line)
# min-max
# score(y) + motif(x) violinplot

# ------

# score hesaplama before-after --> farklı scaling methodlar ve pbm karsılastırma

# gradient descent , L1,L2 kullanalım, logistic regression
# gatavzf ve gata_ctc combine

# sadece log2 target value + log2 + 1 shift
# Elastic penalty deneyelim
# log2 + robust
# sadece log2 without minmax
# mse ve weight değerlerine bakalım

# OLS pbm ve chipseq datası için niye bu kadar farklı sonuc veriyor?
# encode da farklı bir chipseq datası için aynı modellemeleri karşılaştır
# neural network kullanalım mse ve featurelarına bakalım.
# pytorch keras tensorflow


# en az 20 farklı ChIP-seq TF protein'i için modelleme yap
# paired t-test, Wilcoxon signed-rank test
# örnek bir vcf file için t ve p değerlerini hesapla
# 15 Ağustos HIBIT paper submission
#-----------------------------------------------------------------------------------------
# CHECK ENCODE ChIP-seq bednarrowPeak files:
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
def peak_offset(sequence, peak, offset): # +30 / -30 peak offsets
    peak = peak - 1 # peak value exclusive
    left_offset = peak - offset
    right_offset = len(sequence) - (peak + offset)
    if (left_offset) < 0:
        start = 0
        end = (peak + offset + abs(left_offset))
    elif (right_offset) < 0:
        start = (peak - (offset + abs(right_offset)))
        end = (peak + offset)
    else:
        start = (left_offset)
        end = peak + offset
    return sequence[start: end]
def minmax(score):  # Min-max normalization
    if score.max() == 0:
        return score
    else:
        diff_range =  score.max() - score.min()
        minmax_norm = (score - score.min()) / diff_range
    return minmax_norm
def print_full(x): # print binary-based sequences
    y = pd.DataFrame(x)
    y.index = [bio.itoseq(x) for x in y.index]
    y["revcomp"] = [bio.revcompstr(x) for x in y.index]
    return y
def read_chip(pbm_format,norm_method,kmer=6): # reading PBM input and apply transformation to scores and binary seq
    global nonrev_list
    nonrev_list = bio.gen_nonreversed_kmer(kmer) # 2080 features (6-mer DNA)
    pbm_format.columns = ['score','sequence'] #score,sequence ..
    score = norm_method(pbm_format['score']) # log transformation for fluorescent signals
    seqbin = [bio.seqtoi(x) for x in pbm_format['sequence']] #  PBM içindeki her bir sekansı binary gösterimine çevirir
    oligfreq = bio.nonr_olig_freq(seqbin,kmer,nonrev_list) # feature vs sekans içeren count table oluşturur
    return pd.concat([score,oligfreq],axis=1)
def log2trans(score):  # binary logarithmic scaling
    log_scores = np.log2(score)
    return log_scores
def bedtotrainset(filebed,filefasta,ENCODE_ID,TF):

    with open(filefasta) as f:  # fasta file to extract sequences
        fasta = f.readlines()
    locs = [y.split()[0].replace(">", "").replace("(+)", "") for x, y in enumerate(fasta) if x % 2 == 0]
    seqs = [y.rstrip().upper().replace("N", "") for x, y in enumerate(fasta) if x % 2 != 0]

    bed_data = pd.read_csv(filebed, sep="\t",
                           header=None)  # bed narrowpeak file to extract scores
    bed_data.columns = ["chrom", "chromStart", "chromEnd", "name", "score", "strand", "signalValue", "pValue", "qValue",
                        "peak"]
    chipseq_train = pd.DataFrame(
        {"region": locs, "sequence": seqs, "score": bed_data["signalValue"], "peak": bed_data["peak"],
         "Score":bed_data["score"],"qValue": bed_data["qValue"]})
    chipseq_train["peak_seq"] = chipseq_train.apply(lambda x: peak_offset(x["sequence"], x["peak"], 18), axis=1)
    # chipseq_train = chipseq_train[chipseq_train["qValue"] >= 1.3]
    # chipseq_train.reset_index(drop="index",inplace=True)
    pbm_format = pd.DataFrame({0: chipseq_train["score"], 1: chipseq_train["peak_seq"]})
    pbm_format.sort_values(by=0,ascending=False,inplace=True)
    pbm_format.to_csv(f"outputs/ChIPseq_{ENCODE_ID}_{TF}.txt", header=None, index=False, sep="\t")
    return pbm_format

def bedtotrainset2(filebed,filefasta,ENCODE_ID,TF):
    with open(filefasta) as f:  # fasta file to extract sequences
        fasta = f.readlines()
    locs = [y.split()[0].replace(">", "").replace("(+)", "") for x, y in enumerate(fasta) if x % 2 == 0]
    seqs = [y.rstrip().upper().replace("N", "") for x, y in enumerate(fasta) if x % 2 != 0]
    bed_data = pd.read_csv(filebed, sep="\t",
                           header=None)  # bed narrowpeak file to extract scores
    bed_data.columns = ["chrom", "chromStart", "chromEnd", "name", "score", "strand", "peak", "peakend", "rgb"]
    chipseq_train = pd.DataFrame(
        {"region": locs, "sequence": seqs, "score": bed_data["score"], "peak": bed_data["peak"]})
    chipseq_train["peak_seq"] = chipseq_train.apply(lambda x: peak_offset(x["sequence"], x["peak"], 30), axis=1)
    pbm_format = pd.DataFrame({0: chipseq_train["score"], 1: chipseq_train["peak_seq"]})
    pbm_format.sort_values(by=0,ascending=False,inplace=True)
    pbm_format.to_csv(f"outputs/ChIPseq_{ENCODE_ID}_{TF}.txt", header=None, index=False, sep="\t")
    return pbm_format
def TF_bednarrow_to_fasta(bed_file,genome,ENCODE_ID): # fasta conversion
    ref_genome = pyfastx.Fasta(f"genome_assembly/{genome}.fa.gz")
    with open(bed_file, "r") as bed_data, open(f"outputs/{ENCODE_ID}_fasta.txt", "w") as output_file:
        for line in bed_data:
            chrom, start, end = line.strip().split("\t")[:3]
            start = int(start)
            end = int(end)
            sequence = ref_genome[chrom][start:end]
            output_file.write(f">{chrom}:{start}-{end}\n")
            output_file.write(str(sequence) + "\n")
        print(f"{ENCODE_ID} is converted to fasta format!")

def apply_sgd(df,scaler,regularizer,ENCODE_ID):
    X = df.drop('score',axis=1).apply(scaler,axis=0) # values of features
    y = df["score"] # target values
    sgd = SGDRegressor(loss= "squared_error",alpha=0.0001, max_iter=1000, tol=1e-3, penalty=regularizer, eta0=0.1, random_state=333)
    sgd.fit(X, y)
    cov = get_cov_params(sgd, X, y)
    params = sgd.coef_
    print_motif = pd.DataFrame({"Weights": sgd.coef_}, index=nonrev_list)  # for array-like output of OLS result
    print_motif = print_full(print_motif["Weights"].sort_values(ascending=False))
    # print_motif.to_csv(f"outputs/motifs_{ENCODE_ID}.csv")
    return sgd,params,print_motif,cov # [1] = coefficients [-1] = covariance
def mypredict(seq1, seq2, k,params,cov_matrix):
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

def train_ChIP(bed_path, genome,reg, kmer, ENCODE_ID,TF): # MODEL OUTPUT / provides estimate of weights / covariance mat
    TF_bednarrow_to_fasta(bed_path, genome, ENCODE_ID) # write fasta file in your working directory
    pbm_format = bedtotrainset(bed_path, f"outputs/{ENCODE_ID}_fasta.txt", ENCODE_ID,TF) # trainset
    df_chip = read_chip(pbm_format,log2trans,kmer) # frequency table
    sgd_chip_none = apply_sgd(df_chip, minmax, reg,ENCODE_ID) # run model
    with open(f"parameters_{ENCODE_ID}.pkl", "wb") as file:
        pickle.dump(sgd_chip_none, file)
    return df_chip, sgd_chip_none # 4 variables

def train_ChIP2(bed_path, genome, ENCODE_ID,TF): # MODEL OUTPUT / provides estimate of weights / covariance mat
    TF_bednarrow_to_fasta(bed_path, genome, ENCODE_ID) # write fasta file in your working directory
    pbm_format = bedtotrainset2(bed_path, f"outputs/{ENCODE_ID}_fasta.txt", ENCODE_ID,TF) # trainset
    df_chip = read_chip(pbm_format,log2trans) # frequency table
    sgd_chip_none = apply_sgd(df_chip, minmax, "squared_error", None,ENCODE_ID) # run model
    with open(f"parameters_{ENCODE_ID}.pkl", "wb") as file:
        pickle.dump(sgd_chip_none, file)
    return sgd_chip_none # 4 variables

def df_ChIP(bed_path, genome,kmer, ENCODE_ID,TF): # MODEL OUTPUT / provides estimate of weights / covariance mat
    TF_bednarrow_to_fasta(bed_path, genome, ENCODE_ID) # write fasta file in your working directory
    pbm_format = bedtotrainset(bed_path, f"outputs/{ENCODE_ID}_fasta.txt", ENCODE_ID,TF) # trainset
    df_chip = read_chip(pbm_format,log2trans,kmer) # frequency table
    return df_chip,pbm_format

def df_fromPBM(pbm, kmer): # MODEL OUTPUT / provides estimate of weights / covariance mat
    pbm_format = pd.read_csv(pbm, sep="\t", header=None)
    df_chip = read_chip(pbm_format,logshift,kmer) # frequency table
    return df_chip

df,a = df_ChIP("TFs/GATA1/ENCFF853VZF.bed", "hg38",6, "EN", "GATA1")

X = df.drop('score',axis=1).apply(minmax,axis=0) # values of features
y = df["score"] # target values
X = df.drop('score',axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state = 1234)

df_pbm = df_fromPBM("Homo_sapiens_NA_Unpublished_GATA1.txt",6)
df = df_pbm
X_pbm = df_pbm.drop('score',axis=1).apply(minmax,axis=0) # values of features
y_pbm = df_pbm["score"] # target values
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)
ols_pbm = sm.OLS(Y_train, X_train).fit()
ols_pbm = sm.OLS(y, X).fit()
ols_pbm.bic
ols_pbm.summary()
param_pbm = print_full(ols_pbm.params.sort_values(ascending=False))
y_pred = ols_pbm.predict(X_test)
mean_squared_error(Y_test,y_pred)
r2_score(Y_test,y_pred)

# Cross-validation
sg_cross = cross_validate(sgd, X_train, Y_train, scoring="neg_root_mean_squared_error", return_estimator=True,n_jobs=-1)
pb_cross = cross_validate(sgd, X_pbm, y_pbm, scoring="neg_root_mean_squared_error", return_estimator=True,n_jobs=-1)
print(sg_cross["test_score"].mean())
print(pb_cross["test_score"].mean())
param_pbm.columns = ["Weights","revcomp"]

sgd = SGDRegressor(loss= "squared_error",alpha=0.0001, max_iter=1000, tol=1e-3, penalty=None, eta0=0.1, random_state=333)
sgd.fit(X_train, Y_train)
sgd.fit(X, y)

params = sgd.coef_
print_motif = pd.DataFrame({"Weights": sgd.coef_}, index=nonrev_list)  # for array-like output of OLS result
print_motif = print_full(print_motif["Weights"].sort_values(ascending=False))
print(print_motif)
y_pred = sgd.predict(X)
mean_squared_error(y,y_pred)
r2_score(y,y_pred)

y_pred = sgd.predict(X_test)
mean_squared_error(Y_test,y_pred)
r2_score(Y_test,y_pred)
scipy.stats.pearsonr(Y_test,y_pred)[0]

pred2 = (np.dot(X, params) + sgd.intercept_[0])
print(np.allclose(y_pred, pred2))
cov = get_cov_params(sgd,X,y)

cov = get_cov_params(sgd,X_train,Y_train)
seq1 = "ATGATCAGTCTTTGCTCAGTTCTATAAAGAGAAGCAGGATTGTATCCAAGTGTCAACCTA".upper() #  8.279763929029813e-20
seq2 = "ATGATCAGTCTTTGCTCAGTTCTATAAAGATAAGCAGGATTGTATCCAAGTGTCAACCTA".upper()
mypredict(seq1,seq2,6, sgd.coef_,cov)
list(print_motif["Weights"].values[-6:])
list(print_motif.index.values[-6:])


# Mini batch and partial fit
from sklearn.utils import shuffle
X_train, y_train = shuffle(X, y, random_state=42)  #shuffle the data
batch_size = 64
num_samples = len(X)
for i in range(0, num_samples, batch_size):
    X_batch = X[i:i+batch_size]
    y_batch = y[i:i+batch_size]
    sgd.partial_fit(X_batch, y_batch)

# Store coefficients for each run
all_coefficients = []
all_intercept = []

for i in range(100):
    # Create and fit the SGDRegressor model
    sgd = SGDRegressor(loss= "squared_error",alpha=0.0001, max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
    sgd.fit(X_train, Y_train)
    print(f"fit: {i}")
    # Get the coefficients
    coefficients = sgd.coef_
    all_intercept.append(sgd.intercept_)
    # Append the coefficients to the list
    all_coefficients.append(coefficients)

# Convert the list to a NumPy array for easier manipulation
all_coefficients = np.array(all_coefficients)
# Calculate the average coefficients
average_coefficients = np.mean(all_coefficients, axis=0)
avg_intercept = np.mean(all_intercept)
print_motif = pd.DataFrame({"Weights": average_coefficients}, index=nonrev_list)  # for array-like output of OLS result
print_motif = print_full(print_motif["Weights"].sort_values(ascending=False))
print(print_motif)
y_pred = np.dot(X_test, average_coefficients) + avg_intercept
mean_squared_error(Y_test,y_pred)
r2_score(Y_test,y_pred)
scipy.stats.pearsonr(Y_test,y_pred)[0]
#-----------------------------------------------------------------------------------------

# GridSearchCV : check parameters
#
# penalty_grid = [None,"l1","l2"]
# tolerance_grid = [0.001,0.0001]
# alpha_grid = [0.001,0.0001,0.00001]
# eta0_grid = [0.1,0.01]
#
# parameters = {"penalty": penalty_grid,
#               "alpha":alpha_grid,
#               "tol":tolerance_grid,
#               "eta0" : eta0_grid}
# gridCV = GridSearchCV(SGDRegressor(),
#                       param_grid=parameters,
#                       n_jobs=-1)
#
# gridCV.fit(X, y)
#
# best_penalty = gridCV.best_params_['penalty']
# best_tolerance = gridCV.best_params_['tol']
# best_alpha = gridCV.best_params_['alpha']
# best_eta0 = gridCV.best_params_["eta0"]
#
# sgd = SGDRegressor(loss= "squared_error",alpha=best_alpha, learning_rate="invscaling",max_iter=1000,
#                    tol=best_tolerance, penalty=best_penalty, eta0=0.1, random_state=333)

