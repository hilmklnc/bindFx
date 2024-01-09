import os
import numpy as np
import pandas as pd
# import pyfastx
# import pyfaidx
import scipy.stats
from sklearn.linear_model import SGDRegressor
import bio
import pickle
import glob
# functions
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
# from pyfaidx import Fasta
# convert zip file to bgzip : https://github.com/slowkow/pytabix or using the bgzip tool from samtools (htslib).
# pyfaids only supports for compressed bgzip file
# import pyfaidx
# ref_genome = Fasta(f"genome_assembly/hg38.zip",sequence_always_upper=True) # 1-based coordinates
# sequence = ref_genome["chrom"][start-1:end].seq
def TF_bednarrow_to_fasta(bed_file,genome,ENCODE_ID): # fasta conversion
    ref_genome = pyfastx.Fasta(f"genome_assembly/{genome}.fa.gz") # 0-based coordinates
    with open(bed_file, "r") as bed_data, open(f"outputs/encode_fasta/{ENCODE_ID}_fasta.txt", "w") as output_file:
        for line in bed_data:
            chrom, start, end = line.strip().split("\t")[:3]
            start = int(start)
            end = int(end)
            sequence = ref_genome[chrom][start:end]
            output_file.write(f">{chrom}:{start}-{end}\n")
            output_file.write(str(sequence) + "\n")
        print(f"{ENCODE_ID} is converted to fasta format!")
def bedtotrainset(filebed,filefasta,ENCODE_ID,TF,biosample):

    with open(filefasta) as f:  # fasta file to extract sequences
        fasta = f.readlines()
    locs = [y.split()[0].replace(">", "").replace("(+)", "") for x, y in enumerate(fasta) if x % 2 == 0]
    seqs = [y.rstrip().upper().replace("N", "") for x, y in enumerate(fasta) if x % 2 != 0]

    bed_data = pd.read_csv(filebed, sep="\t",
                           header=None)  # bed narrowpeak file to extract scores
    bed_data.columns = ["chrom", "chromStart", "chromEnd", "name", "score", "strand", "signalValue", "pValue", "qValue",
                        "peak"]
    chipseq_train = pd.DataFrame(
        {"region": locs, "sequence": seqs, "score": bed_data["signalValue"], "peak": bed_data["peak"]})
    chipseq_train["peak_seq"] = chipseq_train.apply(lambda x: peak_offset(x["sequence"], x["peak"], 30), axis=1)
    pbm_format = pd.DataFrame({0: chipseq_train["score"], 1: chipseq_train["peak_seq"]})
    pbm_format.sort_values(by=0,ascending=False,inplace=True)
    pbm_format.to_csv(f"outputs/encode_trainsets/ChIPseq_{ENCODE_ID}_{biosample}_{TF}.txt", header=None, index=False, sep="\t")
    return pbm_format
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
    return scaled_cov_matrix
def apply_sgd(df,ENCODE_ID):
    X = df.drop('score',axis=1).apply(minmax,axis=0) # values of features
    y = df["score"] # target values
    sgd = SGDRegressor(loss= "squared_error",alpha=0.0001, max_iter=1000, tol=1e-3, penalty=None, eta0=0.1, random_state=25)
    sgd.fit(X, y)
    cov = get_cov_params(sgd, X, y)
    params = sgd.coef_
    print_motif = pd.DataFrame({"Weights": sgd.coef_}, index=nonrev_list)  # for array-like output of OLS result
    print_motif = print_full(print_motif["Weights"].sort_values(ascending=False))
    # print_motif.to_csv(f"outputs/motifs_{ENCODE_ID}.csv")
    return sgd,params,print_motif,cov # [1] = coefficients [-1] = covariance
def mypredict(seq1, seq2, params,cov_matrix):
    ref = bio.nonr_olig_freq([bio.seqtoi(seq1)],6,nonrev_list)  # from N
    mut = bio.nonr_olig_freq([bio.seqtoi(seq2)],6,nonrev_list)  # to N
    diff_count = mut - ref # c': count matrix
    diff = np.dot(diff_count, params) # c'Bhat --> the difference in binding affinity
    print("\ndiff score aka (wildBeta-mutatedBeta) (c'): ",diff)
    SE = np.sqrt((np.dot(diff_count,cov_matrix) * diff_count).sum(axis=1)) # 2080*2080 X 2080*1 = 2080*1 # a scalar value Standart error
    t =  diff/ SE # t-statistic : c'Bhat / sqrt(c'*covBhat*c)
    p_val = scipy.stats.norm.sf(abs(t))*2 # follows t-distribution
    statdict = {"diff":diff[0], "t":t[0], "p_value":p_val[0] }
    return statdict

def train_ChIP(bed_path, genome, ENCODE_ID,TF): # MODEL OUTPUT / provides estimate of weights / covariance mat
    TF_bednarrow_to_fasta(bed_path, genome, ENCODE_ID) # write fasta file in your working directory
    pbm_format = bedtotrainset(bed_path, f"outputs/{ENCODE_ID}_fasta.txt", ENCODE_ID,TF) # trainset
    df_chip = read_chip(pbm_format,log2trans) # frequency table
    sgd_chip_none = apply_sgd(df_chip, minmax, "squared_error", None,ENCODE_ID) # run model
    with open(f"parameters_{ENCODE_ID}.pkl", "wb") as file:
        pickle.dump(sgd_chip_none, file)
    return sgd_chip_none # 4 variables

def save_params(TF_trainset):
    pbm_format = pd.read_csv(TF_trainset, sep="\t", header=None)
    ENCODE_ID = TF_trainset.split("_")[1]
    TF_name = TF_trainset.split("_")[2].split(".")[0]
    df_chip = read_chip(pbm_format,log2trans) # frequency table
    sgd_chip_none = apply_sgd(df_chip, ENCODE_ID)  # run model
    with open(f"outputs/params_{ENCODE_ID}_{TF_name}.pkl", "wb") as file:
        pickle.dump(sgd_chip_none, file)
    print(f"outputs/{ENCODE_ID}_{TF_name} is trained and saved!")
    return sgd_chip_none
def pred_vep(model,ENCODE_ID):
    stat_df = vep_seq.apply(lambda x: mypredict(x["sequence"],x["altered_seq"],model[1],model[-1]),
                                                   axis=1,result_type="expand")
    pred_df = pd.concat([vep_seq,stat_df],axis=1)
    pred_df.to_csv(f"outputs/pred_{ENCODE_ID}.csv", index=False)
    pred_significant = pred_df[pred_df["p_value"] < 0.05]
    pred_significant.to_csv(f"outputs/sign_pred_{ENCODE_ID}.csv", index=False)
    return pred_df, pred_significant

def pred_mutation(fasta_path, bed_path, ENCODE_ID): # prediction of VEP file
    pbm_format = bedtotrainset(fasta_path, bed_path, ENCODE_ID)
    df_chip = read_chip(pbm_format, log2trans)  # count table of trainset
    sgd_chip_none = apply_sgd(df_chip, minmax, "squared_error", None,ENCODE_ID)
    pred_sgd_chip_none,pred_sign = pred_vep(sgd_chip_none,ENCODE_ID)
    return pred_sgd_chip_none,pred_sign, sgd_chip_none

def prep_train_ChIP(bed_path, genome, ENCODE_ID,TF,biosample): # MODEL OUTPUT / provides estimate of weights / covariance mat
    TF_bednarrow_to_fasta(bed_path, genome, ENCODE_ID) # write fasta file in your working directory
    pbm_format = bedtotrainset(bed_path, f"outputs/encode_fasta/{ENCODE_ID}_fasta.txt", ENCODE_ID,TF,biosample) # trainset
    num_peak = len(pbm_format)
    return num_peak # 4 variables


# TRANSFORM INTO TRAINSET
meta = pd.read_csv("metadata.tsv",sep="\t")
meta["File accession"].nunique()
meta = meta[["File accession","Experiment target","File assembly","Biosample type","Biosample term name","Experiment target"]]
raw_peaks = glob.glob("outputs/ENCODE_2558_beds/*.bed")[100:110]

perf_df = pd.DataFrame(columns=["TF_Name","#peaks","Biosample","Biosample type"])
for raw in raw_peaks:
    ENCODE_ID = raw.split("\\")[1].split(".")[0]
    TF = meta[meta["File accession"] == ENCODE_ID]["Experiment target"].values[0][0].split("-")[0]
    assemble_type = meta[meta["File accession"] == ENCODE_ID]["File assembly"].values[0]
    biosample = meta[meta["File accession"] == ENCODE_ID]["Biosample term name"].values[0]
    biosample_type = meta[meta["File accession"] == ENCODE_ID]["Biosample type"].values[0]
    if "38" in assemble_type:
        genome = "hg38"
    else:
        genome = "hg19"
    print("ID:",ENCODE_ID,"TF:",TF,"hg:",assemble_type, "Biosample:",biosample)
    peaks = prep_train_ChIP(raw,genome,ENCODE_ID,TF,biosample)
    new_df = pd.DataFrame([[TF,peaks,biosample,biosample_type]], index=[ENCODE_ID], columns=["TF_Name","#peaks","Biosample","Biosample type"])
    perf_df = pd.concat([perf_df, new_df])
    print("\n")


