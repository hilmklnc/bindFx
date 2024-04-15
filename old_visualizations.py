# Performance metrics
import os
from sklearn.linear_model import SGDRegressor
import biocoder
import numpy as np
import pandas as pd
import glob
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_circles
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
import scipy

# -----------------------------------BENCHMARKING PERFORMANCE METRICS SGD / OLS-----------------------------------
# ols = pd.read_csv("outputs/Performance_OLS_pbm_minmax.csv")
# sgd = pd.read_csv("outputs/sgd_pbm_summay_L1_2.csv")
#
# sgd["Model"] = "SGD"
# ols["Model"] = "OLS"
# sgd["TFs"] = None
# ols["TFs"] = None
#
# tf_list = glob.glob("TFs_trainset/*.txt")
# tf_list = glob.glob("pbm_trainset_v1/*.txt")
#
# TF_list = []
# for tf in tf_list:
#     TF_list.append(tf.split("_")[-1].split(".")[0])
#
# for i,tf in enumerate(TF_list):
#     sgd["TFs"][i] = tf
#     ols["TFs"][i] = tf
#
# model_merged = sgd.merge(ols,how="outer")
#
# # Mean Squared Error vs TFs
# plt.figure(figsize=(20, 13))
# sns.barplot(data= model_merged, x="TFs",y="mse", hue="Model")
# # Show the plot
# plt.xticks(rotation=90)
# plt.ylabel("MSE",fontsize=25)
# plt.xlabel("TFs",fontsize=25)
# plt.title("SGD vs OLS performance (k=6)",fontsize=30)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.legend(fontsize=15)
# plt.show()
#
# plt.figure(figsize=(20, 13))
# sns.barplot(data= model_merged, x="TFs",y="r", hue="Model")
# plt.xticks(rotation=90)
# plt.ylabel("Pearson Correlation Coefficient",fontsize=25)
# plt.xlabel("TFs",fontsize=25)
# plt.title("MiniBatch-GD vs OLS performance (k=6)",fontsize=30)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.legend(fontsize=15)
# plt.show()
#
#
# plt.figure(figsize=(20, 13))
# sns.barplot(data= model_merged, x="TFs",y="r2", hue="Model")
# plt.xticks(rotation=90)
# plt.ylabel("R2",fontsize=25)
# plt.xlabel("TFs",fontsize=25)
# plt.title("SGD vs OLS performance (k=6)",fontsize=30)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.legend(fontsize=15)
# plt.show()
#
#
# plt.figure(figsize=(22, 25))
# sns.barplot(data= model_merged, y="TFs",x="elapsed_time", hue="Model",orient="h")
# plt.xlabel("Elapsed Time (seconds)",fontsize=30)
# plt.ylabel("TF Models",fontsize=30)
# plt.title("MiniBatch-GD vs OLS performance",fontsize=35)
# plt.xticks(fontsize=25)
# plt.yticks(fontsize=25)
# plt.legend(fontsize=25)
# plt.show()
#
# plt.figure(figsize=(20, 13))
# sns.barplot(data= model_merged, x="TFs",y="medae", hue="Model")
# plt.xticks(rotation=90)
# plt.ylabel("MEDAE",fontsize=25)
# plt.xlabel("TFs",fontsize=25)
# plt.title("SGD vs OLS performance (k=6)",fontsize=30)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.legend(fontsize=15)
# plt.show()
#
# metric_ols = ols.iloc[:,1:-2]
# metric_sgd = sgd.iloc[:,1:-2]
# mean1 = pd.DataFrame(metric_sgd.mean())
# mean2 = pd.DataFrame(metric_ols.mean())
# means = pd.concat([mean1,mean2],axis=1)
# means.columns = ["SGD","OLS"]
# means = means.reset_index()
# means.rename(columns={"index":"Metric"},inplace=True)
# df = means
# bar_width = 0.35 # Set the width of the bars
# # Set the positions of the bars on the x-axis
# index = np.arange(len(df))
# # Create a figure and axis
# fig, ax = plt.subplots()
# # Plot the bars for SGD and OLS
# for model in ['SGD', 'OLS']:
#     ax.bar(index + (df.columns.get_loc(model) - 1) * bar_width, df[model], bar_width, label=model)
# # Set the x-axis labels
# ax.set_xlabel('Metric')
# ax.set_xticks(index)
# ax.set_xticklabels(df['Metric'])
# # Set the y-axis label
# ax.set_ylabel('Average Metric Value for 50 TFs')
# # Add a legend
# ax.legend()
# # Set the title
# ax.set_title('Comparison of 7-mer SGD Model')
# # Show the plot
# plt.yticks(list(range(0,46,5)))
# plt.tight_layout()
# plt.show()

#------------------------------------ BREAST CANCER 560 Analysis of Prediction Results ---------------------------
df_results = pd.read_parquet("outputs/BreastCancer560_results_0.01_probs.pqt")

tf_list = glob.glob("data/trainsets403/*.txt")
TF_list = []
for tf in tf_list:
    TF_list.append(tf.split("_")[-1].split(".")[0])

# TF COUNTS
def pred_TF_counts(df_results,TF_dict_gain,TF_dict_loss):
    # TF_dict_gain = {}
    # TF_dict_loss = {}
    for i, j in zip(df_results["TF_loss_detail"].values, df_results["TF_gain_detail"].values):
        for k in eval(i):
            TF = k[0]
            if TF in TF_dict_loss:
                TF_dict_loss[TF] += 1
            else:
                print(f"{TF} firstly encountered!")
                TF_dict_loss[TF] = 1
        print("------------------######## First step is done ! ########## -------------------------- ")
        for k in eval(j):
            TF = k[0]
            if TF in TF_dict_gain:
                TF_dict_gain[TF] += 1
            else:
                print(f"{TF} firstly encountered!")
                TF_dict_gain[TF] = 1
    # for x in TF_list:
    #     if x not in TF_dict_loss:
    #         print("No Encounter TF: ",x)
    #         TF_dict_loss[x] = 0
    #     if x not in TF_dict_gain:
    #         print("No Encounter TF: ",x)
    #         TF_dict_gain[x] = 0

    return TF_dict_gain,TF_dict_loss

TF_dict_gain, TF_dict_loss = pred_TF_counts(df_results,TF_list)

df_gain = pd.DataFrame([TF_dict_gain])
df_loss = pd.DataFrame([TF_dict_loss])
df_merged_tf = pd.concat([df_gain,df_loss]).T.reset_index()
df_merged_tf.columns = ["TF_name","#Gain","#Loss"]
df_merged_tf.to_csv("counts_TFs_0.01.csv",index=False)

#---------------------------------------------------------------------
df_merged = pd.read_csv("outputs/df_all_stats.csv")

df_merged.sort_values(by="#Gain", ascending=False,inplace=True)
# Count plot
plt.figure(figsize=(50, 55))
sns.set_style('darkgrid') # bottom= instead of left
plt.barh(df_merged["TF_name"],df_merged["#Gain"],label="TF Gain",color="tab:red") # bottom: gain of TF # Gain of Binding
plt.barh(df_merged["TF_name"],df_merged["#Loss"],left=df_merged["#Gain"],label="TF Loss",color="tab:blue") # top: loss of TF #Gain of Binding
plt.xlabel('Count of TFs',fontsize=30)
plt.ylabel('Transcription Factors',fontsize=5)
plt.title(f'All Substitutions',fontsize=50)
plt.legend(fontsize=30,title="Behavior")
plt.xticks(fontsize=35)
plt.yticks(fontsize=30)
# plt.savefig(f'counts_allsubs.png')
plt.show()


# Stacked barplot (Sorted percentages of TFs)
df_merged["Percent_gain"] = (df_merged["#Gain"] * 100) / len(aggregate[aggregate[sbs_name] >= 0.5])
df_merged["Percent_loss"] = (df_merged["#Loss"] * 100) / len(aggregate[aggregate[sbs_name] >= 0.5])
df_merged["Percent_gain"] = (df_merged["#Gain"] * 100) / len(df_results)
df_merged["Percent_loss"] = (df_merged["#Loss"] * 100) / len(df_results)
# df_merged.sort_values(by="Percent_gain",ascending=False,inplace=True)
# df_merged.to_csv("counts_TFs_0.05.csv",index=False)

plt.figure(figsize=(20, 12))
sns.set_style('darkgrid')
plt.bar(df_merged["TF name"],df_merged["Percent_gain"],label="TF Gain") # bottom: gain of TF # Gain of Binding
plt.bar(df_merged["TF name"],df_merged["Percent_loss"],bottom=df_merged["Percent_gain"],label="TF Loss") # top: loss of TF #Gain of Binding
plt.ylabel('Percentage of TFs (%)',fontsize=30)
plt.xlabel('Transcription Factors',fontsize=30)
# plt.title('All Substitutions')
plt.title(f'Percentages of 50 TFs Loss/Gain for {sbs_name}-Specific',fontsize=30)
plt.legend(fontsize=20,title="Behavior")
plt.xticks(rotation=90)
plt.show()

# FOLD CHANGE Plots

#FC graph:
# df_merged["FC"] = (df_merged["#Gain"].replace(0,1) / df_merged["#Loss"].replace(0,1))
df_merged["FC"] = (df_merged["#Gain"] / df_merged["#Loss"])
df_merged.sort_values(by="FC",ascending=False,inplace=True)
df_merged["log2FC"] = np.log2((df_merged["#Gain"] ) / (df_merged["#Loss"] ))
df_merged.to_csv(f"outputs/df_all_stats.csv",index=False)


g = sum(df_merged["FC"]>1)
e = sum(df_merged["FC"] == 1)
l = sum(df_merged["FC"]<1)

plt.figure(figsize=(20, 12))
sns.set_style('darkgrid')
col = (["tab:red"]*g) + (["tab:gray"]*e) + (["tab:blue"]*l)
plt.bar(df_merged["TF name"], df_merged["FC"],color = col) # bottom: gain of TF # Gain of Binding
plt.ylabel('Fold Change (TF gain/TF loss)',fontsize=20)
plt.xlabel('Transcription Factors',fontsize=20)
plt.title(f'Fold Change of TF Loss/Gain for {sbs_name}-Specific',fontsize=30)
plt.xticks(rotation=90)
plt.show()

# Log2(FC):
c = 0.00001
df_merged["log2FC"] = np.log2((df_merged["#Gain"] + c) / (df_merged["#Loss"] + c))
df_merged.to_csv(f"df_{sbs_name}.csv",index=False)
# cmap = plt.get_cmap("viridis")
# color=cmap(np.linspace(0, 1, len(df_FC)))
plt.figure(figsize=(20, 12))
sns.set_style('darkgrid')
plt.bar(df_merged["TF name"], df_merged["log2FC"],color = col) # bottom: gain of TF # Gain of Binding
plt.ylabel('Log2 Fold Change (TF gain/TF loss)',fontsize=20)
plt.xlabel('Transcription Factors',fontsize=20)
plt.title(f'Log2 Fold Change of TF Gain/Loss for {sbs_name}-Specific',fontsize=30)
plt.xticks(rotation=90)
plt.show()

df_merged["FC"] = 0
df_merged["log2FC"] = 0
for i in range(len(df_merged)):
    if ((df_merged["#Gain"].values[i] == 0) or (df_merged["#Loss"].values[i] == 0)):
        FC = (df_merged["#Gain"].iloc[i] + 1) / (df_merged["#Loss"].iloc[i] + 1)
        df_merged["FC"].iloc[i] = FC
        df_merged["log2FC"].iloc[i] = np.log2(FC)
    else:
        FC = (df_merged["#Gain"].iloc[i]) / (df_merged["#Loss"].iloc[i])
        df_merged["FC"].iloc[i] = FC
        df_merged["log2FC"].iloc[i] = np.log2(FC)
df_merged.sort_values(by="FC", inplace=True)
# df_merged = df_merged[(df_merged["log2FC"] > 0.15) | (df_merged["log2FC"] < -0.15)]
df_merged.to_csv(f"df_all_subs.csv", index=False)
g = sum(df_merged["FC"] > 1)
e = sum(df_merged["FC"] == 1)
l = sum(df_merged["FC"] < 1)


# FC graph:
plt.figure(figsize=(20, 25))
sns.set_style('darkgrid')
col = ((["tab:blue"] * l + (["tab:gray"] * e) + ["tab:red"] * g))
plt.barh(df_merged["TF name"], df_merged["FC"], color=col)  # bottom: gain of TF # Gain of Binding
plt.ylabel('Transcription Factors', fontsize=30)
plt.xlabel('Fold Change (TF gain/TF loss)', fontsize=30)
plt.title('All Substitution', fontsize=50)
plt.xticks(fontsize=35)
plt.yticks(fontsize=30)
plt.savefig(f'FC_all_subs.png')
plt.show()

# Log2(FC):
plt.figure(figsize=(20, 25))
sns.set_style('darkgrid')
plt.barh(df_merged["TF name"], df_merged["log2FC"], color=col)  # bottom: gain of TF # Gain of Binding
plt.xlabel('Log2 Fold Change (TF gain/TF loss)', fontsize=30)
plt.ylabel('Transcription Factors', fontsize=30)
plt.title('All Substitution', fontsize=50)
plt.xticks(fontsize=35)
plt.yticks(fontsize=30)
plt.savefig('log2FC_all_subs.png')
plt.show()

#------------------------------------------------------------------------------
# SIGNATURE SPECIFIC PLOTS

#--------------------------------------------------------------------------------------------------------
# function:
# aggregate = pd.read_csv("breastcancer_21VCF/aggregate_preds.csv")
tf_list = glob.glob("TFs_trainset/*.txt")
df_results = pd.read_parquet("outputs/BreastCancer560_results_0.01_probs.pqt")

num_of_gof_lof = 10

sbs_id = "SBS1"
min_number = 10
aggregate = df_results
def sbs_plots(aggregate,sbs_id,min_number,tf_list):
    TF_list = []
    for tf in tf_list:
        TF_list.append(tf.split("_")[-1].split(".")[0])
    TF_dict_gain = {tf:0 for tf in TF_list}
    TF_dict_loss = {tf:0 for tf in TF_list}
    num_of_gof_lof = min_number
    sbs_name = sbs_id
    df_sbs = aggregate[aggregate[sbs_name] >= 0.5]
    TF_dict_gain, TF_dict_loss = pred_TF_counts(df_sbs,TF_dict_gain,TF_dict_loss)
    df_merged = pd.DataFrame([TF_dict_gain,TF_dict_loss]).T.reset_index()
    df_merged.columns = ["TF name","#Gain","#Loss"]
    # aggregate[aggregate[sbs_name] >= 0.5].apply(count_TFs,axis=1)
    df_merged = df_merged[(df_merged["#Gain"] >= num_of_gof_lof) | (df_merged["#Loss"] >= num_of_gof_lof)]
    df_merged.sort_values(by="#Gain", ascending=False, inplace=True)
    df_merged["Percent_gain"] = (df_merged["#Gain"] * 100) / len(aggregate[aggregate[sbs_name] >= 0.5])
    df_merged["Percent_loss"] = (df_merged["#Loss"] * 100) / len(aggregate[aggregate[sbs_name] >= 0.5])
    #----- Count plot
    # plt.figure(figsize=(18, 15))
    # sns.set_style('darkgrid')
    # plt.bar(df_merged["TF name"],df_merged["#Gain"],label="TF Gain",color="tab:red") # bottom: gain of TF # Gain of Binding
    # plt.bar(df_merged["TF name"],df_merged["#Loss"],bottom=df_merged["#Gain"],label="TF Loss",color="tab:blue") # top: loss of TF #Gain of Binding
    # plt.ylabel('Count of TFs',fontsize=20)
    # plt.xlabel('Transcription Factors',fontsize=20)
    # plt.title(f'{sbs_name}',fontsize=30)
    # plt.legend(fontsize=20,title="Behavior")
    # plt.xticks(fontsize=18,rotation=90)
    # plt.savefig(f'{sbs_name}/counts_{sbs_name}.png')
    # plt.show()
    # PERCENTAGES
    # plt.figure(figsize=(18, 15))
    # sns.set_style('darkgrid')
    # plt.bar(df_merged["TF name"],df_merged["Percent_gain"],label="TF Gain",color="tab:red") # bottom: gain of TF # Gain of Binding
    # plt.bar(df_merged["TF name"],df_merged["Percent_loss"],bottom=df_merged["Percent_gain"],label="TF Loss",color="tab:blue") # top: loss of TF #Gain of Binding
    # plt.ylabel('Percentage of TFs (%)',fontsize=20)
    # plt.xlabel('Transcription Factors',fontsize=20)
    # # plt.title('All Substitutions')
    # plt.title(f'{sbs_name}',fontsize=30)
    # plt.legend(fontsize=20,title="Behavior")
    # plt.xticks(fontsize=18,rotation=90)
    # plt.savefig(f'{sbs_name}/percentages_{sbs_name}.png')
    # plt.show()
    # FOLD CHANGE Calculations
    # df_merged["#Gain"].replace(0,1,inplace=True)
    # df_merged["#Loss"].replace(0,1,inplace=True)
    df_merged["FC"] = 0
    df_merged["log2FC"] = 0
    for i in range(len(df_merged)):
        if ((df_merged["#Gain"].values[i] == 0) or (df_merged["#Loss"].values[i] == 0)):
            FC = (df_merged["#Gain"].iloc[i] + 1) / (df_merged["#Loss"].iloc[i] + 1)
            df_merged["FC"].iloc[i] = FC
            df_merged["log2FC"].iloc[i] = np.log2(FC)
        else:
            FC = (df_merged["#Gain"].iloc[i]) / (df_merged["#Loss"].iloc[i])
            df_merged["FC"].iloc[i] = FC
            df_merged["log2FC"].iloc[i]  = np.log2(FC)
    df_merged.sort_values(by="FC",inplace=True)
    # df_merged = df_merged[(df_merged["log2FC"] > 0.15) | (df_merged["log2FC"] < -0.15)]
    # df_merged.to_csv(f"{sbs_name}/df_{sbs_name}.csv",index=False)
    g = sum(df_merged["FC"] > 1)
    e = sum(df_merged["FC"] == 1)
    l = sum(df_merged["FC"] < 1)

    #FC graph:
    # plt.figure(figsize=(20,20))
    # sns.set_style('darkgrid')
    # col = (["tab:red"]*g) + (["tab:gray"]*e) + (["tab:blue"]*l)
    # plt.barh(df_merged["TF name"], df_merged["FC"],color = col) # bottom: gain of TF # Gain of Binding
    # plt.ylabel('Transcription Factors',fontsize=30)
    # plt.yticks(fontsize=30)
    # plt.xlabel('Fold Change (TF gain/TF loss)',fontsize=30)
    # plt.title(f'{sbs_name}',fontsize=50)
    # plt.xticks(fontsize=35)
    # # plt.savefig(f'{sbs_name}/FC_{sbs_name}.png')
    # plt.show()

    # Log2(FC):
    # plt.figure(figsize=(18, 15))
    # sns.set_style('darkgrid')
    # plt.bar(df_merged["TF name"], df_merged["log2FC"],color = col) # bottom: gain of TF # Gain of Binding
    # plt.ylabel('Log2 Fold Change (TF gain/TF loss)',fontsize=20)
    # plt.xlabel('Transcription Factors',fontsize=20)
    # plt.title(f'{sbs_name}',fontsize=30)
    # plt.xticks(fontsize=18,rotation=90)
    # plt.savefig(f'{sbs_name}/log2FC_{sbs_name}.png')
    # plt.show()

SBSs = ["SBS1","SBS2","SBS3","SBS5","SBS13","SBS40"]
min_number = 10
for x in SBSs:
    sbs_plots(x,num_of_gof_lof)
#------------------------------------------------------------------------------------------------------
# VENN DIAGRAM BETWEEN SBSs
aggregate = pd.read_csv("breastcancer_21VCF/aggregate_preds.csv")

tf_list = glob.glob("TFs_trainset/*.txt")
def count_TFs(row):
    for i in range(len(eval(row["TF_gain_detail"]))):
        TF = eval(row["TF_gain_detail"])[i][0]
        if TF in TF_dict_gain:
            TF_dict_gain[TF] += 1
        else:
            print(f"{TF} firstly encountered!")
            TF_dict_gain[TF] = 1
    for i in range(len(eval(row["TF_loss_detail"]))):
        TF = eval(row["TF_loss_detail"])[i][0]
        if TF in TF_dict_loss:
            TF_dict_loss[TF] += 1
        else:
            TF_dict_loss[TF] = 1
def common_TF_venn(sbs_id,min_number):
    global TF_dict_gain,TF_dict_loss
    TF_list = []
    for tf in tf_list:
        TF_list.append(tf.split("_")[-1].split(".")[0])

    TF_dict_gain = {tf:0 for tf in TF_list}
    TF_dict_loss = {tf:0 for tf in TF_list}
    num_of_gof_lof = min_number
    sbs_name = sbs_id
    aggregate[aggregate[sbs_name] >= 0.5].apply(count_TFs,axis=1)
    df_merged = pd.DataFrame([TF_dict_gain,TF_dict_loss]).T.reset_index()
    df_merged.columns = ["TF name","#Gain","#Loss"]
    df_merged = df_merged[(df_merged["#Gain"] >= num_of_gof_lof) | (df_merged["#Loss"] >= num_of_gof_lof)]
    df_merged.reset_index(drop="index",inplace=True)
    return df_merged


SBSs = ["SBS1","SBS2","SBS3","SBS5","SBS13","SBS40"]
min_number = 10
dict_gain_tfs = {}
dict_loss_tfs = {}
for sbs in SBSs:
    df = common_TF_venn(sbs,min_number)
    # ind_g = df["#Gain"] > min_number
    # ind_l = df["#Loss"] > min_number
    dict_gain_tfs[sbs] = df[df["#Gain"] > df["#Loss"]]["TF name"]
    dict_loss_tfs[sbs] = df[df["#Gain"] < df["#Loss"]]["TF name"]
dict_gain_tfs["SBS2"]
dict_gain_tfs["SBS13"]
# ind_sbs1_g = df_dicts["SBS1"]["#Gain"] > min_number
# ind_sbs2_g = df_dicts["SBS2"]["#Gain"] > min_number
#
# ind_sbs1_l = df_dicts["SBS1"]["#Loss"] > min_number
# ind_sbs2_l = df_dicts["SBS2"]["#Loss"] > min_number
#
# #GOF
# set1 = set(df_dicts["SBS1"][ind_sbs1_g]["TF name"])
# set2 = set(df_dicts["SBS5"][ind_sbs2_g]["TF name"])
#
# #LOF
# set1 = set(df_dicts["SBS1"][ind_sbs1_l]["TF name"])
# set2 = set(df_dicts["SBS5"][ind_sbs2_l]["TF name"])

# GOF
for x in combinations(dict_gain_tfs,2):
    plt.figure(figsize=(18, 15))
    set1 = set(dict_gain_tfs[x[0]])
    set2 = set(dict_gain_tfs[x[1]])
    venn = venn2([set1, set2], set_labels=(f"{x[0]}",f"{x[1]}"),set_colors=("salmon", "tomato"),alpha=0.7)
    for text in venn.set_labels:
        text.set_fontsize(30)
    for text in venn.subset_labels:
        text.set_fontsize(23)
    if len(set1.intersection(set2)) == 0:
        venn.get_label_by_id('10').set_text('\n'.join(set1 - set2))
        venn.get_label_by_id('01').set_text('\n'.join(set2 - set1))
        plt.title(f"Venn Diagram of {x[0]} and {x[1]} - (GOF)", fontsize=30)
        plt.savefig(f'venn_diags/LOF/{x[0]}_{x[1]}_lof.png')
        plt.show()
    else:
        venn.get_label_by_id('10').set_text('\n'.join(set1 - set2))
        venn.get_label_by_id('11').set_text('\n'.join(map(str, set1.intersection(set2))))
        venn.get_label_by_id('01').set_text('\n'.join(set2 - set1))
        venn.get_patch_by_id('11').set_color('red')
        plt.title(f"Venn Diagram of {x[0]} and {x[1]} - (GOF)", fontsize=30)
        plt.savefig(f'venn_diags/GOF/{x[0]}_{x[1]}_gof.png')
        plt.show()

# LOF
for x in combinations(dict_loss_tfs,2):
    plt.figure(figsize=(18, 15))
    set1 = set(dict_loss_tfs[x[0]])
    set2 = set(dict_loss_tfs[x[1]])
    venn = venn2([set1, set2], set_labels=(f"{x[0]}",f"{x[1]}"),set_colors=("cornflowerblue","royalblue"),alpha=0.7)
    for text in venn.set_labels:
        text.set_fontsize(30)
    if len(set1.intersection(set2)) == 0:
        venn.get_label_by_id('10').set_text('\n'.join(set1 - set2))
        venn.get_label_by_id('01').set_text('\n'.join(set2 - set1))
        venn.get_label_by_id('10').set_fontsize(22)
        venn.get_label_by_id('01').set_fontsize(22)
        plt.title(f"Venn Diagram of {x[0]} and {x[1]} - (LOF)", fontsize=30)
        plt.savefig(f'venn_diags/LOF/{x[0]}_{x[1]}_lof.png')
        plt.show()
    else:
        for text in venn.subset_labels:
            text.set_fontsize(23)
        venn.get_label_by_id('10').set_text('\n'.join(set1 - set2))
        venn.get_label_by_id('11').set_text('\n'.join(map(str, set1.intersection(set2))))
        venn.get_label_by_id('01').set_text('\n'.join(set2 - set1))
        venn.get_patch_by_id('11').set_color('blue')
        plt.title(f"Venn Diagram of {x[0]} and {x[1]} - (LOF)", fontsize=30)
        plt.savefig(f'venn_diags/LOF/{x[0]}_{x[1]}_lof.png')
        plt.show()


a = "SBS1"
b = "SBS3"
plt.figure(figsize=(18, 15))
set1 = set(dict_loss_tfs[a])
set2 = set(dict_loss_tfs[b])
venn = venn2([set1, set2], set_labels=(a,b),set_colors=("salmon", "tomato"),alpha=0.7)
for text in venn.set_labels:
    text.set_fontsize(30)
for text in venn.subset_labels:
    text.set_fontsize(23)
venn.get_label_by_id('10').set_text('\n'.join(set1-set2))
venn.get_label_by_id('11').set_text('\n'.join(map(str, set1.intersection(set2))))
venn.get_label_by_id('01').set_text('\n'.join(set2-set1))
# venn.get_label_by_id('10').set_fontsize(22)
# venn.get_label_by_id('01').set_fontsize(22)
venn.get_patch_by_id('11').set_color('red')
plt.title(f"Venn Diagram of SBS3 and SBS40 - (GOF)", fontsize=30)
plt.savefig(f'venn_diags/GOF/{x[0]}_{x[1]}_gof.png')
plt.show()

#-------------------------------------------------------------------------


# y axis TFs
def sbs_plots_yaxis(sbs_id,min_number):
    TF_list = []
    for tf in tf_list:
        TF_list.append(tf.split("_")[-1].split(".")[0])

    TF_dict_gain = {tf:0 for tf in TF_list}
    TF_dict_loss = {tf:0 for tf in TF_list}
    num_of_gof_lof = min_number
    sbs_name = sbs_id

    aggregate[aggregate[sbs_name] >= 0.5].apply(count_TFs,axis=1)
    df_merged = pd.DataFrame([TF_dict_gain,TF_dict_loss]).T.reset_index()
    df_merged.columns = ["TF name","#Gain","#Loss"]
    df_merged = df_merged[(df_merged["#Gain"] >= num_of_gof_lof) | (df_merged["#Loss"] >= num_of_gof_lof)]
    df_merged.sort_values(by="#Gain", ascending=False, inplace=True)
    df_merged["Percent_gain"] = (df_merged["#Gain"] * 100) / len(aggregate[aggregate[sbs_name] >= 0.5])
    df_merged["Percent_loss"] = (df_merged["#Loss"] * 100) / len(aggregate[aggregate[sbs_name] >= 0.5])
    # Count plot
    plt.figure(figsize=(18, 15))
    sns.set_style('darkgrid')
    plt.bar(df_merged["TF name"],df_merged["#Gain"],label="TF Gain",color="tab:red") # bottom: gain of TF # Gain of Binding
    plt.bar(df_merged["TF name"],df_merged["#Loss"],bottom=df_merged["#Gain"],label="TF Loss",color="tab:blue") # top: loss of TF #Gain of Binding
    plt.ylabel('Count of TFs',fontsize=20)
    plt.xlabel('Transcription Factors',fontsize=20)
    plt.title(f'{sbs_name}',fontsize=30)
    plt.legend(fontsize=20,title="Behavior")
    plt.xticks(fontsize=18,rotation=90)
    # plt.savefig(f'{sbs_name}/counts_{sbs_name}.png')
    plt.show()
    # PERCENTAGES
    plt.figure(figsize=(18, 15))
    sns.set_style('darkgrid')
    plt.bar(df_merged["TF name"],df_merged["Percent_gain"],label="TF Gain",color="tab:red") # bottom: gain of TF # Gain of Binding
    plt.bar(df_merged["TF name"],df_merged["Percent_loss"],bottom=df_merged["Percent_gain"],label="TF Loss",color="tab:blue") # top: loss of TF #Gain of Binding
    plt.ylabel('Percentage of TFs (%)',fontsize=20)
    plt.xlabel('Transcription Factors',fontsize=20)
    # plt.title('All Substitutions')
    plt.title(f'{sbs_name}',fontsize=30)
    plt.legend(fontsize=20,title="Behavior")
    plt.xticks(fontsize=18,rotation=90)
    plt.savefig(f'{sbs_name}/percentages_{sbs_name}.png')
    plt.show()
    # FOLD CHANGE Calculations
    # df_merged["#Gain"].replace(0,1,inplace=True)
    # df_merged["#Loss"].replace(0,1,inplace=True)
    df_merged["FC"] = 0
    df_merged["log2FC"] = 0
    for i in range(len(df_merged)):
        if ((df_merged["#Gain"].values[i] == 0) or (df_merged["#Loss"].values[i] == 0)):
            FC = (df_merged["#Gain"].iloc[i] + 1) / (df_merged["#Loss"].iloc[i] + 1)
            df_merged["FC"].iloc[i] = FC
            df_merged["log2FC"].iloc[i] = np.log2(FC)
        else:
            FC = (df_merged["#Gain"].iloc[i]) / (df_merged["#Loss"].iloc[i])
            df_merged["FC"].iloc[i] = FC
            df_merged["log2FC"].iloc[i]  = np.log2(FC)
    df_merged.sort_values(by="FC",inplace=True)
    # df_merged = df_merged[(df_merged["log2FC"] > 0.15) | (df_merged["log2FC"] < -0.15)]
    # df_merged.to_csv(f"{sbs_name}/df_{sbs_name}.csv",index=False)
    g = sum(df_merged["FC"] > 1)
    e = sum(df_merged["FC"] == 1)
    l = sum(df_merged["FC"] < 1)

    #FC graph:
    plt.figure(figsize=(20,20))
    sns.set_style('darkgrid')
    col = ( (["tab:blue"]*l+ (["tab:gray"]*e) + ["tab:red"]*g))
    plt.barh(df_merged["TF name"], df_merged["FC"],color = col) # bottom: gain of TF # Gain of Binding
    plt.ylabel('Transcription Factors',fontsize=30)
    plt.yticks(fontsize=30)
    plt.xlabel('Fold Change (TF gain/TF loss)',fontsize=30)
    plt.title(f'{sbs_name}',fontsize=50)
    plt.xticks(fontsize=35)
    plt.savefig(f'{sbs_name}/FC_{sbs_name}.png')
    plt.show()

    # Log2(FC):
    plt.figure(figsize=(20,20))
    sns.set_style('darkgrid')
    plt.barh(df_merged["TF name"], df_merged["log2FC"],color = col) # bottom: gain of TF # Gain of Binding
    plt.xlabel('Log2 Fold Change (TF gain/TF loss)',fontsize=30)
    plt.ylabel('Transcription Factors',fontsize=30)
    plt.title(f'{sbs_name}',fontsize=50)
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=30)
    plt.savefig(f'{sbs_name}/log2FC_{sbs_name}.png')
    plt.show()

tfdata = pd.read_csv("TF_Information.txt",sep="\t")
print(tfdata[tfdata["Motif_Type"] == "PBM"].TF_Name[:123])
tfdata.to_csv("TF_info.csv")

tf_data_pbm = tfdata[tfdata["Motif_Type"] == "PBM"]
tf_data_pbm["Motif_ID"].nunique() # 165 different TF
tf_data_pbm["TF_Name"].nunique() # 165 different TF
tf_data_pbm.to_csv("TF_PBM_info.csv")



# boxplot representation for finding outliers of gain/loss counts in SBS-specific
# def low_up_outliers(df_sbs,tag,whis=3):
#     Q1 = df_sbs[tag].quantile(q=0.25)
#     Q3 = df_sbs[tag].quantile(q=0.75)
#     IQR = Q3 - Q1
#     low_out = Q1 - (IQR*whis)
#     up_out = Q3 + (IQR*whis)
#     return low_out,up_out
# def outlier_df(df_sbs):
#     low_out_gain, up_out_gain = low_up_outliers(df_sbs,"#Gain")
#     low_out_loss, up_out_loss = low_up_outliers(df_sbs, "#Loss")
#
#     df_filt_gain = df_sbs.loc[(df_sbs["#Gain"] > low_out_gain) & (df_sbs["#Gain"] < up_out_gain)]
#     df_outlier_gain = df_sbs.loc[(df_sbs["#Gain"] >= up_out_gain) | (df_sbs["#Gain"] <= low_out_gain)]
#
#     df_filt_loss = df_sbs.loc[(df_sbs["#Loss"] > low_out_loss) & (df_sbs["#Loss"] < up_out_loss)]
#     df_outlier_loss = df_sbs.loc[(df_sbs["#Loss"] >= up_out_loss) | (df_sbs["#Loss"] <= low_out_loss)]
#
#     return df_filt_gain,df_outlier_gain, df_filt_loss,df_outlier_loss
#
# df_filt_gain,df_outlier_gain, df_filt_loss,df_outlier_loss = outlier_df(df_sbs13)
# def plot_sbs_outliers(df_sbs,sbs_name):
#     if not os.path.exists(f"outputs/{sbs_name}"):
#         os.makedirs(f"outputs/{sbs_name}")
#     df_filt_gain, df_outlier_gain, df_filt_loss, df_outlier_loss = outlier_df(df_sbs)
#     df_count_max = np.max(df_sbs.loc[:, ["#Gain", "#Loss"]])
#     if df_count_max > 20000 :
#         step = 10000
#     else:
#         step = 1000
#     # Plotting BOXPLOT
#     plt.figure(figsize=(8, 10))
#     sns.boxplot(df_sbs.loc[:, ["#Gain", "#Loss"]], whis=3)
#     plt.yticks(np.arange(0, df_count_max, step=step))
#     plt.title(f"{sbs_name}")
#     plt.ylabel("Significant Counts")
#     plt.savefig(f'outputs/{sbs_name}/{sbs_name}_boxplot.png')
#     plt.show()
#
#     # Plotting scatterplot with Gain and Loss
#     plt.figure(figsize=(8, 6))
#     # Plot Gain and Loss points
#     plt.scatter(df_filt_gain.index, df_filt_gain['#Gain'], color="tab:red", label="Gain", marker='.')
#     plt.scatter(df_filt_loss.index, df_filt_loss['#Loss'], color="tab:blue", label='Loss', marker='.')
#     # Plot outliers
#     plt.scatter(df_outlier_gain.index, df_outlier_gain['#Gain'], color="tab:red", marker='x', label='Gain Outliers')
#     plt.scatter(df_outlier_loss.index, df_outlier_loss['#Loss'], color="tab:blue", marker='x', label='Loss Outliers')
#     plt.xlabel('TFs')
#     plt.ylabel('Significant Counts')
#     plt.title(f"{sbs_name}")
#     plt.legend()
#     plt.savefig(f'outputs/{sbs_name}/{sbs_name}_scatter.png')
#     plt.show()
#     print(f"{sbs_name}: , plotted!")
#
# sbs_files = glob.glob("outputs/*filtered_*.csv")
#
# for sbs_file in sbs_files:
#     sbs_name = sbs_file.split("_")[-1].split(".")[0]
#     df_sbs = pd.read_csv(sbs_file)
#     plot_sbs_outliers(df_sbs,sbs_name)
