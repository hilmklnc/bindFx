import glob
import pandas as pd
import os
import pickle
import biocoder
import scipy.stats
import numpy as np
import statsmodels.stats.multitest as mlt
import warnings
warnings.filterwarnings("ignore")



# Merge all chunks into single TF prediction file
def merge_chunks(files_path=f"truba_results/pred_results/chunk#{i}/*.pqt",num_chunk=177):
    pred_TF_files = {}
    num_chunks = num_chunk
    for i in range(num_chunks): # store each chunks for corresponding TF
        pred_results = glob.glob(files_path)
        for j in pred_results:
            TF_name = j.split("_")[-1].split(".")[0]
            if TF_name not in pred_TF_files.keys():
                pred_TF_files[TF_name] = [j]
            else:
                pred_TF_files[TF_name].append(j)
    for k,v in pred_TF_files.items():
        df_list = [pd.read_parquet(file_path) for file_path in v]
        combined_df = pd.concat(df_list,ignore_index=True)
        combined_df.to_parquet(f"outputs/all_preds/pred_BreastCancer560_{k}.pqt", index=False)
        print("TF:",k,"Saved!")

# check for any errors
ls = glob.glob("truba_results/preds_BreastCancer560/*.pqt")

total_rows = 3479651
for x in ls:
    df = pd.read_parquet(ls)
    a = len(df)
    if a != total_rows:
        print("-----------------SMTHNG GOING WRONG----------------!!!!!")
        print(ls)
    elif a == total_rows:
        print(a)
    else:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

#########################################################################################

# MULTIPLE CORRECTION TESTING
pred_files = glob.glob("truba_results/preds_BreastCancer560/*.pqt") # 21 vcf seq files
# os.makedirs(f"truba_results/preds_BreastCancer560_qval")
def FDR_control(pred_files):
    for i,pred_file in enumerate(pred_files):
        pred_file_ID = pred_file.split("\\")[1]
        pred_data = pd.read_parquet(pred_file)
        pred_data["adj_pvalue"] = mlt.multipletests(pred_data["p_value"], method='bonferroni')[1]
        pred_data.to_parquet(f"truba_results/preds_BreastCancer560_qval/mlt_{pred_file_ID}",index=False)
        print(i,": Multiple Correction testing is done for", pred_file_ID)

#########################################################################################
#ADDING GAIN/LOSS INFO for all predictions

def mean_update(current_mean, current_count, new_value):
    old_sum = current_mean * current_count
    return abs((old_sum + abs(new_value)) / (current_count + 1))
def gain_or_loss_vectorized(df, pred_vcf, alpha):
    loss_indices = df.index[(pred_vcf["adj_pvalue"] <= alpha) & (pred_vcf["diff"] < 0)]
    gain_indices = df.index[(pred_vcf["adj_pvalue"] <= alpha) & (pred_vcf["diff"] > 0)]
    # Update loss and gain counts
    df.loc[loss_indices, 'TF_loss'] += 1
    df.loc[gain_indices, 'TF_gain'] += 1

    # Update loss and gain diffs
    df.loc[loss_indices, 'TF_loss_diff_mean'] = mean_update(df.loc[loss_indices, 'TF_loss_diff_mean'],
                                                            df.loc[loss_indices, 'TF_loss_detail'].apply(len),
                                                            pred_vcf.loc[loss_indices, 'diff'])
    df.loc[gain_indices, 'TF_gain_diff_mean'] = mean_update(df.loc[gain_indices, 'TF_gain_diff_mean'],
                                                            df.loc[gain_indices, 'TF_gain_detail'].apply(len),
                                                            pred_vcf.loc[gain_indices, 'diff'])

    # Update loss and gain details
    loss_info = pred_vcf.loc[loss_indices, ['TF_name', 'diff', 'adj_pvalue']].values.tolist()
    gain_info = pred_vcf.loc[gain_indices, ['TF_name', 'diff', 'adj_pvalue']].values.tolist()

    for i, loss in zip(loss_indices, loss_info):
        df['TF_loss_detail'].iloc[i].append(loss)
    for j, gain in zip(gain_indices, gain_info):
        df['TF_gain_detail'].iloc[j].append(gain)
    return df

def columnize_preds(alpha=0.01):
    pred_files = glob.glob(f"truba_results/preds_BreastCancer560_qval/*.pqt")
    vcf_seq = pd.read_parquet("main_vcf_seq.pqt")
    vcf_seq["TF_loss"] = 0  # add columns
    vcf_seq["TF_gain"] = 0
    vcf_seq["TF_loss_diff_mean"] = 0
    vcf_seq["TF_gain_diff_mean"] = 0
    vcf_seq["TF_loss_detail"] = [[] for _ in range(len(vcf_seq))]
    vcf_seq["TF_gain_detail"] = [[] for _ in range(len(vcf_seq))]

    for step, pred_path in enumerate(pred_files):
        TF_name = pred_path.split("_")[-1].split(".")[0]
        pred_df = pd.read_parquet(pred_path)
        pred_df["TF_name"] = TF_name
        # Apply vectorized function
        vcf_seq = gain_or_loss_vectorized(vcf_seq, pred_df, alpha)
        print(step, "---- Columnized: ", TF_name)

    vcf_seq.to_parquet(f"outputs/BreastCancer560_loss_gain_results_{alpha}.pqt", index=False)

    return vcf_seq

result = columnize_preds(alpha=0.01)

# result["TF_loss"].sort_values(ascending=False)
# result["TF_gain"].sort_values(ascending=False)
# result["TF_gain_diff_mean"].sort_values(ascending=False)
#
# mn = []
# for x in result["TF_gain_detail"].iloc[1749898]:
#     mn.append(x[1])
# np.mean(mn)
# result["TF_gain_diff_mean"].iloc[1749898]
# result["TF_gain"].iloc[1749898]

df_results = pd.read_parquet("outputs/BreastCancer560_loss_gain_results_0.01.pqt")
df_results.info()
df_results.head()
print(df_results.sort_values(by="TF_loss",ascending=False).iloc[:15,7:11])
print(df_results.sort_values(by="TF_gain",ascending=False).iloc[:15,7:11])

#----------------------------------------------------------------------------------------------

# Check different significance levels for VEP file
# vep_loss_gain_data = glob.glob("vep_loss_gain_data_*.csv")
# mydict = {"TF_Totalcounts":[], "TF_NumofLoss":[], "TF_NumofGain":[], "TF_Loss_AvgWeight":[], "TF_Gain_AvgWeight":[]}
# for vep in vep_loss_gain_data:
#     df = pd.read_csv(vep)
#     mydict["TF_Totalcounts"].extend([df["TF_loss"].sum()+df["TF_gain"].sum()])
#     mydict["TF_NumofLoss"].extend([df[df["driver"] == 1]["TF_loss"].mean() /df[df["driver"] == 0]["TF_loss"].mean()])
#     mydict["TF_NumofGain"].extend([df[df["driver"] == 1]["TF_gain"].mean() / df[df["driver"] == 0]["TF_gain"].mean()])
#     mydict["TF_Loss_AvgWeight"].extend([df[df["driver"] == 1]["TF_loss_diff"].mean() / df[df["driver"] == 0]["TF_loss_diff"].mean()])
#     mydict["TF_Gain_AvgWeight"].extend([df[df["driver"] == 1]["TF_gain_diff"].mean() / df[df["driver"] == 0]["TF_gain_diff"].mean()])
#
# mean_df = pd.DataFrame(mydict,index=vep_loss_gain_data)
#
# mean_df.sort_values(by=["TF_NumofLoss","TF_NumofGain"],ascending=False,inplace=True)
# mean_df.to_csv("driver_vs_nondriver.csv")

#------------------------------------------------------------------------------------


