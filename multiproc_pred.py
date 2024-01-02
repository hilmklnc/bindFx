import glob
import pandas as pd
import os
import pickle
import bio
import scipy.stats
import numpy as np
import multiprocessing
import time
def mypredict(seq1, seq2, params,cov_matrix,TF_name):
    ref = bio.nonr_olig_freq([bio.seqtoi(seq1)],6,nonrev_list)  # from N
    mut = bio.nonr_olig_freq([bio.seqtoi(seq2)],6,nonrev_list)  # to N
    diff_count = mut - ref # c': count matrix
    diff = np.dot(diff_count, params) # c'Bhat --> the difference in binding affinity
    print(f"Process {os.getpid()} and {TF_name} diff score aka (wildBeta-mutatedBeta) (c'): ",diff)
    SE = np.sqrt((np.dot(diff_count,cov_matrix) * diff_count).sum(axis=1)) # 2080*2080 X 2080*1 = 2080*1 # a scalar value Standart error
    t =  diff/ SE # t-statistic : c'Bhat / sqrt(c'*covBhat*c)
    p_val = scipy.stats.norm.sf(abs(t))*2 # follows t-distribution
    statdict = {"diff":diff[0], "t":t[0], "p_value":p_val[0] }
    return statdict

def pred_vcf(param,ENCODE_ID,TF_name,vcf_seq,vcf_ID):
    stat_df = vcf_seq.apply(lambda x: mypredict(x["sequence"],x["altered_seq"],param[1],param[-1],TF_name),
                                                   axis=1,result_type="expand")
    pred_df = pd.concat([vcf_seq,stat_df],axis=1)
    pred_df.to_csv(f"{vcf_ID}/pred_{vcf_ID}_{ENCODE_ID}_{TF_name}.csv", index=False)

def process_vcf(param, ENCODE_ID, TF_name, vcf_seq,vcf_ID):
    start = time.perf_counter()
    print(f"Process {os.getpid()} is analyzing VCF file: {vcf_ID}, ENCODE_ID: {ENCODE_ID}, TF_name: {TF_name}")
    pred_vcf(param, ENCODE_ID, TF_name, vcf_seq,vcf_ID)
    print(f"Process {os.getpid()} finished analyzing VCF file: {vcf_ID}, ENCODE_ID: {ENCODE_ID}, TF_name: {TF_name}")
    end = time.perf_counter()
    print(f"Time -----------:{end-start}")

nonrev_list = bio.gen_nonreversed_kmer(6)  # 2080 features (6-mer DNA)
# Listing pre-computed-pred files
params = glob.glob("params/*.pkl")
vcf_seqs = glob.glob("breastcancer_21VCF/VCF_seq/*.csv")  # I have 21 vcf files to be analyzed
param_dict = {}  # store pre-computed parameters
for param in params:  # I have 50 pre-computed parameters of models
    with open(param, "rb") as file:
        param_dict[param] = pickle.load(file)

if __name__ == "__main__":
    with multiprocessing.Pool() as pool:
        for vcf_seq in vcf_seqs:
            vcf_ID = vcf_seq.split("\\")[1].split("_")[0]
            vcf_seq = pd.read_csv(vcf_seq)

            if not os.path.exists(f"{vcf_ID}"):
                os.makedirs(f"{vcf_ID}")
                print(f"{vcf_ID} folder is created!")

            for param_file, param_data in param_dict.items():
                ENCODE_ID = param_file.split("_")[1]
                TF_name = param_file.split("_")[2].split(".")[0]
                pool.apply_async(process_vcf, args=(param_data, ENCODE_ID, TF_name, vcf_seq, vcf_ID))
        # Close the pool and wait for all tasks to complete
        pool.close()
        pool.join()



