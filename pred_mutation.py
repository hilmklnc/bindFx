import glob
import pandas as pd
import os
import pickle
import bio
import scipy.stats
import numpy as np
import statsmodels.stats.multitest as mlt

def mypredict(seq1, seq2, params,cov_matrix):
    ref = bio.nonr_olig_freq([bio.seqtoi(seq1)],6,nonrev_list)  # from N
    mut = bio.nonr_olig_freq([bio.seqtoi(seq2)],6,nonrev_list)  # to N
    diff_count = mut - ref # c': count matrix
    diff = np.dot(diff_count, params) # c'Bhat --> the difference in binding affinity
    # print("\ndiff score aka (wildBeta-mutatedBeta) (c'): ",diff)
    SE = np.sqrt((np.dot(diff_count,cov_matrix) * diff_count).sum(axis=1)) # 2080*2080 X 2080*1 = 2080*1 # a scalar value Standart error
    t =  diff/ SE # t-statistic : c'Bhat / sqrt(c'*covBhat*c)
    p_val = scipy.stats.norm.sf(abs(t))*2 # follows t-distribution
    statdict = {"diff":diff[0], "t":t[0], "p_value":p_val[0] }
    return statdict

def pred_vcf(param,ENCODE_ID,TF_name,vcf_seq):
    stat_df = vcf_seq.apply(lambda x: mypredict(x["sequence"],x["altered_seq"],param[1],param[-1]),
                                                   axis=1,result_type="expand")
    pred_df = pd.concat([vcf_seq,stat_df],axis=1)
    pred_df.to_csv(f"{vcf_ID}/pred_{vcf_ID}_{ENCODE_ID}_{TF_name}.csv", index=False)

# def pred_mutation(sgd_chip_none,vcf_file, ENCODE_ID): # prediction of VEP file
#     pred_vcf = pred_vcf(sgd_chip_none,vcf_file,ENCODE_ID)
#     return pred_vcf
def mean_update(current_mean, current_count, new_value):
    if new_value < 0:
        updated_sum = (-current_mean) * current_count + new_value
    else:
        updated_sum = current_mean * current_count + new_value
    updated_count = current_count + 1
    updated_mean = updated_sum / updated_count
    return abs(updated_mean)
def gain_or_loss(row,alpha):
    q_values = row['adj_pvalue']
    p_values = row["p_value"]
    diff = row['diff']
    ind = row["index"]
    if q_values <= alpha:
        if diff < 0: # if diff score is negative, then loss of TF exist
            vcf_seq["TF_loss"][ind] += 1
            vcf_seq["TF_loss_diff"][ind]=mean_update(vcf_seq["TF_loss_diff"][ind],len(vcf_seq["TF_loss_detail"][ind]), diff)
            vcf_seq["TF_loss_detail"][ind].append([TF_name, diff, p_values,q_values])
        elif diff > 0: # if diff score is positive, then gain of TF exist
            vcf_seq["TF_gain"][ind] += 1
            vcf_seq["TF_gain_diff"][ind]=mean_update(vcf_seq["TF_gain_diff"][ind],len(vcf_seq["TF_gain_detail"][ind]), diff)
            vcf_seq["TF_gain_detail"][ind].append([TF_name, diff, p_values,q_values])
def addcolumn_gain_loss(vcf_seq_path,pred_vcfs,alpha):
    global vcf_seq, TF_name
    vcf_seq = pd.read_csv(vcf_seq_path)
    vcf_seq["TF_loss"] = 0 # add columns
    vcf_seq["TF_gain"] = 0
    vcf_seq["TF_loss_diff"] = 0
    vcf_seq["TF_gain_diff"] = 0
    vcf_seq["TF_loss_detail"] = [ [] for _ in range(len(vcf_seq))]
    vcf_seq["TF_gain_detail"] = [ [] for _ in range(len(vcf_seq))]
    for pred_vcf in pred_vcfs:
        TF_name = pred_vcf.split("_")[-1].split(".")[0]
        pred_sgd = pd.read_csv(pred_vcf)
        pred_sgd.reset_index(inplace=True)
        pred_sgd.apply(gain_or_loss,alpha=alpha,axis=1)
    vcf_seq.to_csv(f"breastcancer_21VCF/pred_all/{vcf_ID}_loss_gain_data_{alpha}.csv",index=False)

nonrev_list = bio.gen_nonreversed_kmer(6) # 2080 features (6-mer DNA)
#-------------------------------------------------------------------------------------------------------------------
# Creating pre-computed-pred VCF files
params = glob.glob("params/*.pkl")
vcf_seqs = glob.glob("breastcancer_21VCF/VCF_seq_files/*.csv") # I have 21 vcf files to be analyzed
param_dict = {} # store pre-computed parameters
for param in params: # I have 50 pre-computed parameters of models
    with open(param,"rb") as file:
        param_dict[param] = pickle.load(file)

for vcf_seq in vcf_seqs: # for each vcf seq, I will compute their t and p values for each TF models
    vcf_ID = vcf_seq.split("\\")[1].split("_")[0]
    vcf_seq = pd.read_csv(vcf_seq)
    if not os.path.exists(f"{vcf_ID}"):
        os.makedirs(f"{vcf_ID}")
    else:
        continue
    print(f"{vcf_ID} folder is created!")
    for id,param in param_dict.items():
        ENCODE_ID = id.split("_")[1]
        TF_name = id.split("_")[2].split(".")[0]
        pred_vcf(param, ENCODE_ID,TF_name,vcf_seq) # predictions of TF models
    print(f"{vcf_ID} is predicted by SGD TF-models!")
#-------------------------------------------------------------------------------------------------------------------
# ADDING Mutational signature probabilities
def mutation_context(context,alt_point):
    ref_seq = context[1]
    alt_seq = alt_point
    five_prime = context[0]
    three_prime = context[2]
    return five_prime + "[" +  ref_seq + ">" + alt_seq + "]" + three_prime

def add_context(row):
    ref_point = row["sequence"][30]
    alt_point = row["altered_seq"][30]
    if  ref_point == "G" or ref_point == "A":
        contxt = bio.revcompstr(row["sequence"][29:32])
        alt_point = bio.revcompstr(alt_point)
    else:
        contxt = row["sequence"][29:32]
    mutated_context = mutation_context(contxt,alt_point)
    return mutated_context

probs = pd.read_csv("breastcancer_21VCF/sample_probabilities/COSMIC_SBS96_Decomposed_Mutation_Probabilities.txt",sep="\t")

vcf_seqs = glob.glob("breastcancer_21VCF/VCF_seq_files/*.csv") # 21 vcf seq files

for vcf_seq in vcf_seqs:
    vcf = pd.read_csv(vcf_seq)
    col = ["chr", "start", "Sample Names", "ref", "alt", "sequence", "altered_seq"]
    vcf.columns = col
    vcf_ID = vcf["Sample Names"][0]
    vcf["MutationTypes"] = vcf.apply(add_context, axis=1)
    vcf = pd.merge(vcf, probs, how="left")
    vcf.to_csv(f"outputs/{vcf_ID}_seq_probs.csv", index=False)

#-------------------------------------------------------------------------------------------------------------------
vcf_seqs = glob.glob("breastcancer_21VCF/VCF_seq_probs/*.csv") # 21 vcf seq files
folders = os.listdir('breastcancer_21VCF/pred_21VCF_adjusted')# Specify the directory path you want to list

# Multiple Correction Testing:
for folder in folders:
    vcf_ID = folder
    os.makedirs(f"outputs/{vcf_ID}")
    pred_files = glob.glob(f"breastcancer_21VCF/pred_21VCF_raw_pvalues/{vcf_ID}/*.csv")
    for pred_file in pred_files:
        pred_file_ID = pred_file.split("\\")[1]
        pred_data = pd.read_csv(pred_file)
        pred_data["adj_pvalue"] = mlt.multipletests(pred_data["p_value"], method='bonferroni')[1]
        pred_data.to_csv(f"outputs/{vcf_ID}/{pred_file_ID}",index=False)
#-------------------------------------------------------------------------------------------------------------------
# FEATURIZING THE VCF FILE :
# for alpha = 0.05 threshold for q values
for folder,vcf_seq in zip(folders,vcf_seqs):
    vcf_ID = folder
    pred_files = glob.glob(f"breastcancer_21VCF/pred_21VCF_adjusted/{vcf_ID}/*.csv")
    addcolumn_gain_loss(vcf_seq,pred_files,0.05)
    print(f"{folder} file is generated")

# different p values
for x in [0.05,0.01,0.001,0.0001,0.00001]:
    folder = f"pred_21vcf_{x}"
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"pred_21vcf_{x} folder is created!")
    for folder, vcf_seq in zip(folders, vcf_seqs):
        vcf_ID = folder
        pred_files = glob.glob(f"breastcancer_21VCF/pred_21VCF/{vcf_ID}/*.csv")
        addcolumn_gain_loss(vcf_seq, pred_files, x)
        print(f"{folder} file is generated!")
    print(f"predictions having {x} alpha value are completed!")


#----------------------------------------------------------------------------------------------

# Check different significance levels for VEP file

vep_loss_gain_data = glob.glob("vep_loss_gain_data_*.csv")
mydict = {"TF_Totalcounts":[], "TF_NumofLoss":[], "TF_NumofGain":[], "TF_Loss_AvgWeight":[], "TF_Gain_AvgWeight":[]}
for vep in vep_loss_gain_data:
    df = pd.read_csv(vep)
    mydict["TF_Totalcounts"].extend([df["TF_loss"].sum()+df["TF_gain"].sum()])
    mydict["TF_NumofLoss"].extend([df[df["driver"] == 1]["TF_loss"].mean() /df[df["driver"] == 0]["TF_loss"].mean()])
    mydict["TF_NumofGain"].extend([df[df["driver"] == 1]["TF_gain"].mean() / df[df["driver"] == 0]["TF_gain"].mean()])
    mydict["TF_Loss_AvgWeight"].extend([df[df["driver"] == 1]["TF_loss_diff"].mean() / df[df["driver"] == 0]["TF_loss_diff"].mean()])
    mydict["TF_Gain_AvgWeight"].extend([df[df["driver"] == 1]["TF_gain_diff"].mean() / df[df["driver"] == 0]["TF_gain_diff"].mean()])

mean_df = pd.DataFrame(mydict,index=vep_loss_gain_data)

mean_df.sort_values(by=["TF_NumofLoss","TF_NumofGain"],ascending=False,inplace=True)
mean_df.to_csv("driver_vs_nondriver.csv")

#------------------------------------------------------------------------------------

# aggregate all pred files
pred_qval = glob.glob("breastcancer_21VCF/pred_all_qval_0.05/*.csv")
aggregate = pd.DataFrame()
for file in pred_qval:
    df = pd.read_csv(file)
    aggregate = pd.concat([aggregate,df])
aggregate.reset_index(drop="index",inplace=True)
aggregate.to_csv("aggregate_preds.csv",index=False)


#------------------------------------------------------------------------------------
