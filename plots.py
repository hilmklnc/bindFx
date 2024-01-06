# Performance metrics
import os
from sklearn.linear_model import SGDRegressor
import statsmodels.api as sm
import bio
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
def apply_ols(df,ENCODE_ID,TF_name,scaler=minmax):
    X = df.drop('score',axis=1).apply(scaler,axis=0)
    y = df["score"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    start_time = time.perf_counter()
    lm = sm.OLS(Y_train, X_train).fit()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    params = print_full(lm.params.sort_values(ascending=False))
    params.columns = ["Weights", "revcomp"]
    y_pred = lm.predict(X_test)
    mse = mean_squared_error(Y_test, y_pred)
    r = scipy.stats.pearsonr(Y_test,y_pred)[0]
    r2 = r2_score(Y_test,y_pred)
    rmse = mean_squared_error(Y_test, y_pred, squared=False)
    mae = mean_absolute_error(Y_test, y_pred)
    medae = median_absolute_error(Y_test, y_pred)
    n_peaks = len(X)
    top_motifs = params
    # TF_motifs = params.to_csv(f"outputs/TF_motifs_OLS_36/ols_motif_{ENCODE_ID}_{TF_name}.csv")
    top_motifs = list(params.index.values[:6])
    top_weights = params["Weights"].values[:6]
    bottom_motifs = list(params.index.values[-6:])
    bottom_weights = list(params["Weights"].values[-6:])
    return [n_peaks,mse, r, r2, rmse, mae, medae, elapsed_time,top_motifs,top_weights,bottom_motifs,bottom_weights]
def apply_sgd_metrics(df, scaler, lss, regularizer,ENCODE_ID,TF_name):
    X = df.drop('score', axis=1).apply(scaler, axis=0)
    y = df["score"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    sgd = SGDRegressor(loss=lss, alpha=0.0001, max_iter=1000, tol=1e-3, penalty=regularizer, eta0=0.1, random_state=333)
    start_time = time.perf_counter()
    sgd.fit(X_train, Y_train)
    end_time = time.perf_counter()
    print_motif = pd.DataFrame({"Weights": sgd.coef_}, index=nonrev_list)  # for array-like output of OLS result
    print_motif = print_full(print_motif["Weights"].sort_values(ascending=False))
    y_pred = sgd.predict(X_test)
    elapsed_time = end_time - start_time
    mse = mean_squared_error(Y_test, y_pred)
    r = scipy.stats.pearsonr(Y_test, y_pred)[0]
    r2 = r2_score(Y_test, y_pred)
    # Root Mean Squared Error (RMSE)
    rmse = mean_squared_error(Y_test, y_pred, squared=False)
    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(Y_test, y_pred)
    # Median Absolute Error(MEDAE)
    medae = median_absolute_error(Y_test, y_pred)
    n_peaks = len(X)
    TF_motifs = print_motif.to_csv(f"outputs/TF_motifs_SGD_36/sgd_motif_{ENCODE_ID}_{TF_name}.csv")
    top_motifs = list(print_motif.index.values[:6])
    top_weights = print_motif["Weights"].values[:6]
    bottom_motifs = list(print_motif.index.values[-6:])
    bottom_weights = list(print_motif["Weights"].values[-6:])
    return [n_peaks,mse, r, r2, rmse, mae, medae, elapsed_time,top_motifs,top_weights,bottom_motifs,bottom_weights]  # [1] = coefficients [-1] = covariance
def read_chip(pbm_format,norm_method,kmer=6): # reading PBM input and apply transformation to scores and binary seq
    global nonrev_list
    nonrev_list = bio.gen_nonreversed_kmer(kmer) # 2080 features (6-mer DNA)
    pbm_format.columns = ['score','sequence'] #score,sequence ..
    score = norm_method(pbm_format['score']) # log transformation for fluorescent signals
    seqbin = [bio.seqtoi(x) for x in pbm_format['sequence']] #  PBM içindeki her bir sekansı binary gösterimine çevirir
    oligfreq = bio.nonr_olig_freq(seqbin,kmer,nonrev_list) # feature vs sekans içeren count table oluşturur
    return pd.concat([score,oligfreq],axis=1)
def perf_test(train_file,method="sgd"):
    ENCODE_ID = train_file.split("_")[2]
    TF_name = train_file.split("_")[3].split(".")[0]
    pbm_format = pd.read_csv(train_file, sep="\t", header=None)
    # pbm_format[1] = pbm_format[1].apply(variable_seq) # 36 bp
    # print(len(pbm_format[1][0]))
    df_chip = read_chip(pbm_format, log2trans)  # count table of trainset
    if method == "sgd":
        sgd_chip_none = apply_sgd_metrics(df_chip, minmax, "squared_error", None,ENCODE_ID,TF_name)
        return sgd_chip_none
    else:
        ols_chip = apply_ols(df_chip,ENCODE_ID,TF_name)
        return ols_chip
def variable_seq(row):
    row = row[12:48]
    return row
trainsets = glob.glob("TFs_trainset/*.txt")

perf_df = pd.DataFrame(columns=["n_peaks","mse", "r", "r2", "rmse", "mae", "medae", "elapsed_time","top_motifs","top_weights","bottom_motifs","bottom_weights"])
for train_file in trainsets:
    name = train_file.split("\\")[1]
    x = perf_test(train_file,"ols")
    new = pd.DataFrame([x], index=[name], columns=["n_peaks","mse", "r","r2", "rmse", "mae", "medae", "elapsed_time","top_motifs","top_weights","bottom_motifs","bottom_weights"])
    perf_df = pd.concat([perf_df, new])
    print(f"{name} is trained!")

perf_df.to_csv("Performance_OLS_splitset_36.csv")

ols = pd.read_csv("outputs/Performance_OLS_pbm_minmax.csv")
sgd = pd.read_csv("outputs/sgd_pbm_summay_L1_2.csv")

sgd["Model"] = "SGD"
ols["Model"] = "OLS"
sgd["TFs"] = None
ols["TFs"] = None

tf_list = glob.glob("TFs_trainset/*.txt")
tf_list = glob.glob("pbm_trainset_v1/*.txt")

TF_list = []
for tf in tf_list:
    TF_list.append(tf.split("_")[-1].split(".")[0])

for i,tf in enumerate(TF_list):
    sgd["TFs"][i] = tf
    ols["TFs"][i] = tf

model_merged = sgd.merge(ols,how="outer")

# Mean Squared Error vs TFs
plt.figure(figsize=(20, 13))
sns.barplot(data= model_merged, x="TFs",y="mse", hue="Model")
# Show the plot
plt.xticks(rotation=90)
plt.ylabel("MSE",fontsize=25)
plt.xlabel("TFs",fontsize=25)
plt.title("SGD vs OLS performance (k=6)",fontsize=30)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.show()

plt.figure(figsize=(20, 13))
sns.barplot(data= model_merged, x="TFs",y="r", hue="Model")
plt.xticks(rotation=90)
plt.ylabel("Pearson Correlation Coefficient",fontsize=25)
plt.xlabel("TFs",fontsize=25)
plt.title("MiniBatch-GD vs OLS performance (k=6)",fontsize=30)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.show()


plt.figure(figsize=(20, 13))
sns.barplot(data= model_merged, x="TFs",y="r2", hue="Model")
plt.xticks(rotation=90)
plt.ylabel("R2",fontsize=25)
plt.xlabel("TFs",fontsize=25)
plt.title("SGD vs OLS performance (k=6)",fontsize=30)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.show()


plt.figure(figsize=(22, 25))
sns.barplot(data= model_merged, y="TFs",x="elapsed_time", hue="Model",orient="h")
plt.xlabel("Elapsed Time (seconds)",fontsize=30)
plt.ylabel("TF Models",fontsize=30)
plt.title("MiniBatch-GD vs OLS performance",fontsize=35)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=25)
plt.show()

plt.figure(figsize=(20, 13))
sns.barplot(data= model_merged, x="TFs",y="medae", hue="Model")
plt.xticks(rotation=90)
plt.ylabel("MEDAE",fontsize=25)
plt.xlabel("TFs",fontsize=25)
plt.title("SGD vs OLS performance (k=6)",fontsize=30)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.show()



metric_ols = ols.iloc[:,1:-2]
metric_sgd = sgd.iloc[:,1:-2]
mean1 = pd.DataFrame(metric_sgd.mean())
mean2 = pd.DataFrame(metric_ols.mean())
means = pd.concat([mean1,mean2],axis=1)
means.columns = ["SGD","OLS"]
means = means.reset_index()
means.rename(columns={"index":"Metric"},inplace=True)
df = means
bar_width = 0.35 # Set the width of the bars
# Set the positions of the bars on the x-axis
index = np.arange(len(df))
# Create a figure and axis
fig, ax = plt.subplots()
# Plot the bars for SGD and OLS
for model in ['SGD', 'OLS']:
    ax.bar(index + (df.columns.get_loc(model) - 1) * bar_width, df[model], bar_width, label=model)
# Set the x-axis labels
ax.set_xlabel('Metric')
ax.set_xticks(index)
ax.set_xticklabels(df['Metric'])
# Set the y-axis label
ax.set_ylabel('Average Metric Value for 50 TFs')
# Add a legend
ax.legend()
# Set the title
ax.set_title('Comparison of 7-mer SGD Model')
# Show the plot
plt.yticks(list(range(0,46,5)))
plt.tight_layout()
plt.show()

#----------------------------------------------------------------------------------

aggregate = pd.read_csv("breastcancer_21VCF/aggregate_preds.csv")

tf_list = glob.glob("TFs_trainset/*.txt")
TF_list = []
for tf in tf_list:
    TF_list.append(tf.split("_")[-1].split(".")[0])

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

aggregate.apply(count_TFs, axis=1)

df_gain = pd.DataFrame([TF_dict_gain])
df_loss = pd.DataFrame([TF_dict_loss])
df_merged = pd.concat([df_gain,df_loss]).T.reset_index()
df_merged.columns = ["TF name","#Gain","#Loss"]

# df_merged.to_csv("counts_TFs_0.05.csv",index=False)

#---------------------------------------------------------------------
df_merged = pd.read_csv("counts_TFs_0.05.csv")
df_merged.sort_values(by="#Gain",inplace=True)
# Count plot
plt.figure(figsize=(20, 25))
sns.set_style('darkgrid') # bottom= instead of left
plt.barh(df_merged["TF name"],df_merged["#Gain"],label="TF Gain",color="tab:red") # bottom: gain of TF # Gain of Binding
plt.barh(df_merged["TF name"],df_merged["#Loss"],left=df_merged["#Gain"],label="TF Loss",color="tab:blue") # top: loss of TF #Gain of Binding
plt.xlabel('Count of TFs',fontsize=30)
plt.ylabel('Transcription Factors',fontsize=30)
plt.title(f'All Substitutions',fontsize=50)
plt.legend(fontsize=30,title="Behavior")
plt.xticks(fontsize=35)
plt.yticks(fontsize=30)
plt.savefig(f'counts_allsubs.png')
plt.show()


# Stacked barplot (Sorted percentages of TFs)
df_merged["Percent_gain"] = (df_merged["#Gain"] * 100) / len(aggregate[aggregate[sbs_name] >= 0.5])
df_merged["Percent_loss"] = (df_merged["#Loss"] * 100) / len(aggregate[aggregate[sbs_name] >= 0.5])
# df_merged["Percent_gain"] = (df_merged["#Gain"] * 100) / len(aggregate)
# df_merged["Percent_loss"] = (df_merged["#Loss"] * 100) / len(aggregate)
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
df_merged["#Gain"].replace(0,1,inplace=True)
df_merged["#Loss"].replace(0,1,inplace=True)

#FC graph:
df_merged["FC"] = (df_merged["#Gain"] / df_merged["#Loss"])
df_merged.sort_values(by="FC",ascending=False,inplace=True)

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
aggregate = pd.read_csv("breastcancer_21VCF/aggregate_preds.csv")
tf_list = glob.glob("TFs_trainset/*.txt")

num_of_gof_lof = 10

sbs_id = "SBS40"
min_number = 10
def sbs_plots(sbs_id,min_number):
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
    col = (["tab:red"]*g) + (["tab:gray"]*e) + (["tab:blue"]*l)
    plt.barh(df_merged["TF name"], df_merged["FC"],color = col) # bottom: gain of TF # Gain of Binding
    plt.ylabel('Transcription Factors',fontsize=30)
    plt.yticks(fontsize=30)
    plt.xlabel('Fold Change (TF gain/TF loss)',fontsize=30)
    plt.title(f'{sbs_name}',fontsize=50)
    plt.xticks(fontsize=35)
    # plt.savefig(f'{sbs_name}/FC_{sbs_name}.png')
    plt.show()

    # Log2(FC):
    plt.figure(figsize=(18, 15))
    sns.set_style('darkgrid')
    plt.bar(df_merged["TF name"], df_merged["log2FC"],color = col) # bottom: gain of TF # Gain of Binding
    plt.ylabel('Log2 Fold Change (TF gain/TF loss)',fontsize=20)
    plt.xlabel('Transcription Factors',fontsize=20)
    plt.title(f'{sbs_name}',fontsize=30)
    plt.xticks(fontsize=18,rotation=90)
    plt.savefig(f'{sbs_name}/log2FC_{sbs_name}.png')
    plt.show()

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
def sbs_plots(sbs_id,min_number):
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
import pandas as pd
tfdata = pd.read_csv("TF_Information.txt",sep="\t")
print(tfdata[tfdata["Motif_Type"] == "PBM"].TF_Name[:123])
tfdata.to_csv("TF_info.csv")

tf_data_pbm = tfdata[tfdata["Motif_Type"] == "PBM"]
tf_data_pbm["Motif_ID"].nunique() # 165 different TF
tf_data_pbm["TF_Name"].nunique() # 165 different TF
tf_data_pbm.to_csv("TF_PBM_info.csv")
