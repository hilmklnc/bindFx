import os
import random
import glob
import biocoder
import numpy as np
import pandas as pd
import glob
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from scipy.stats import fisher_exact
from scipy.stats import chi2_contingency
import statsmodels.stats.multitest as mlt
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

#------------------------------------ BREAST CANCER 560 Analysis of Prediction Results ---------------------------
df_results = pd.read_parquet("outputs/BreastCancer560_results_0.01_probs.pqt")
df_DBD_family = pd.read_csv("TF_model_family.csv")
TF_files = glob.glob("data/trainsets403/*.txt")

# TF_list = []
# for tf in TF_files:
#     TF_list.append(tf.split("_")[-1].split(".")[0])
# TF_dict_gain = {tf: 0 for tf in TF_list}
# TF_dict_loss = {tf: 0 for tf in TF_list}

#TF COUNTS ON AGGREGATED PREDICTIONS
def pred_TF_counts(df_results,TF_dict_gain,TF_dict_loss):
    step = 0
    for i, j in zip(df_results["TF_loss_detail"].values, df_results["TF_gain_detail"].values):
        for k in eval(i):
            TF = k[0]
            if TF in TF_dict_loss:
                TF_dict_loss[TF] += 1
            else:
                print(f"{TF} firstly encountered!")
                TF_dict_loss[TF] = 1
        for k in eval(j):
            TF = k[0]
            if TF in TF_dict_gain:
                TF_dict_gain[TF] += 1
            else:
                print(f"{TF} firstly encountered!")
                TF_dict_gain[TF] = 1
        step += 1
        print(f"Row(mutation index) number: {step}")
    return TF_dict_gain,TF_dict_loss


# TF_dict_gain, TF_dict_loss = pred_TF_counts(df_results,TF_dict_gain,TF_dict_loss)

# df_gain = pd.DataFrame([TF_dict_gain])
# df_loss = pd.DataFrame([TF_dict_loss])
# df_merged_tf = pd.concat([df_gain,df_loss]).T.reset_index()
# df_merged_tf.columns = ["TF_name","#Gain","#Loss"]
# df_merged["Percent_gain"] = (df_merged["#Gain"] * 100) / len(df_results)
# df_merged["Percent_loss"] = (df_merged["#Loss"] * 100) / len(df_results)
# df_merged["FC"] = (df_merged["#Gain"] / df_merged["#Loss"])
# df_merged.sort_values(by="FC",ascending=False,inplace=True)
# df_merged["log2FC"] = np.log2((df_merged["#Gain"] ) / (df_merged["#Loss"] ))
# df_merged.to_csv(f"outputs/df_all_stats.csv",index=False)


# ------------------------------SBS-SPECIFIC ANALYSIS-----------------------------------------
# sbs_id = "SBS1"
# min_number = 10
# aggregate = df_results


# Statistic Tests
def apply_fisher_pvalue(row):
    g = row["#Gain"]
    l = row["#Loss"]
    half_total_count = (g+ l)/2
    expected_gain,expected_loss = half_total_count,half_total_count
    contingency_table = [[g, l], [expected_gain, expected_loss]]
    # print(pd.DataFrame(contingency_table))
    odds_ratio, p_value = fisher_exact(contingency_table, alternative="two-sided")
    # print(f"Odds Ratio: {odds_ratio}")
    # print(f"P-value: {p_value}")
    return  p_value

# def apply_chi2_pvalue(row):
#     g = row["#Gain"]
#     l = row["#Loss"]
#     if (g == 0) & (l == 0):
#         return 1
#     else:
#         half_total_count = (g+ l)/2
#         expected_gain,expected_loss = half_total_count,half_total_count
#         contingency_table = [[g, l], [expected_gain, expected_loss]]
#         print(pd.DataFrame(contingency_table))
#         p_value = chi2_contingency(contingency_table)[1]
#         print(f"p-value: {p_value}")
#         return  p_value

def fish_exact(df_sbs,sbs_name,df_DBD_family,log2FC_limit,alpha=0.05):
    dbd_names = []
    [dbd_names.extend(df_DBD_family[df_DBD_family["TF_Name"] == tf]["DBD_family"].values) for tf in df_sbs["TF name"]]
    df_sbs["DBD_family"] = dbd_names
    df_sbs["p_value_fisher"] = df_sbs.apply(apply_fisher_pvalue,axis=1)
    df_sbs["q_value_fisher"] = mlt.multipletests(df_sbs["p_value_fisher"],method='bonferroni')[1]
    # df_sbs["chi2_p_value"] = df_sbs.apply(apply_chi2_pvalue,axis=1)
    # df_sbs["chi2_adj_pvalue2"] = mlt.multipletests(df_sbs["chi2_p_value"],method='bonferroni')[1]
    pvalue_threshold = alpha
    gain_FC_threshold = 2**(log2FC_limit)
    loss_FC_threshold = 2**(-log2FC_limit)
    effects = np.where((df_sbs["FC"] > gain_FC_threshold) & (df_sbs['q_value_fisher'] < pvalue_threshold), 'GOF',   # gof (significant)
                      np.where((df_sbs['FC'] < loss_FC_threshold) & (df_sbs['q_value_fisher'] < pvalue_threshold), 'LOF',  # lof (significant)
                               'Neutral'))  # Not significant
    df_sbs["TF_binding"] = effects
    print(f"Mutational Signature: {sbs_name}, SBS-specific results are featurized! ")
    return df_sbs

df_p = pd.read_csv("outputs/df_stats_0.75_0.25/dfstats_SBS2_0.75.csv")

df_p = pd.read_csv("outputs/df_stats_0.75_0.25/dfstats_SBS13_0.75.csv")

sbs_name = "SBS13"

def plot_volcano(df_p,sbs_name,prob,log2FC_limit,alpha=0.05):
    min_p_value = df_p["q_value_fisher"][df_p["q_value_fisher"] > 0].min() * 10e-2
    df_p["nonzero_adjpvalue"] = df_p["q_value_fisher"].replace(0,min_p_value)
    df_p["-log10adj-pval"] = -np.log10(df_p["nonzero_adjpvalue"])
    pvalue_threshold = alpha
    logp_threshold = -np.log10(alpha)
    gain_FC_threshold = 2**(log2FC_limit)
    loss_FC_threshold = 2**(-log2FC_limit)
    x_range = max(abs(df_p["log2FC"]. min()),abs(df_p["log2FC"].max()))
    df_p["label"] = df_p["TF_binding"].replace({"GOF":"tab:red", "LOF":"tab:blue", "Neutral":"gray"})
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=df_p,x="log2FC",y = "-log10adj-pval",color=df_p["label"].values ,alpha=0.8,marker="o")
    for i in range(df_p.shape[0]):
        if df_p['DBD_family'].iloc[i] == "Forkhead": #Forkhead, Homeodomain; POU, Myb/SANT
    #         if df_p["-log10adj-pval"].iloc[i] < -np.log10(min_p_value):
            plt.text(df_p['log2FC'].iloc[i], df_p['-log10adj-pval'].iloc[i], df_p['TF name'].iloc[i],
                         fontsize=10, rotation = 60)
    gof_count = sum(df_p['label'] == 'tab:red')
    lof_count = sum(df_p['label'] == 'tab:blue')
    ns_count = sum(df_p['label'] == 'gray')

    plt.ylabel("-log10(adj_pvalue)")
    plt.xlabel("log2FC")
    plt.axhline(y=logp_threshold, color='grey', linestyle='--')
    plt.axhline(y=-np.log10(min_p_value),color="red",linestyle='--')
    plt.axvline(x=np.log2(gain_FC_threshold), color='grey',linestyle='--')
    plt.axvline(x=np.log2(loss_FC_threshold), color='grey',linestyle='--')
    plt.xlim(-(x_range+1),(x_range+1))
    # markers = {df_p[ df_p["DBD_family"] != "Myb/SANT"]: ".", df_p[ df_p["DBD_family"] == "Myb/SANT"]: "v"}
    legend_labels = [f"GOF ({gof_count})", f"LOF ({lof_count})", f"Not Significant ({ns_count})"]
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10)
               for color, label in zip(['tab:red', 'tab:blue', 'gray'], legend_labels)]
    plt.legend(handles=handles,loc = "upper left",framealpha=0.1)
    plt.title(sbs_name)
    print(f"Mutational Signature: {sbs_name}, volcano plot is done! ")
    # plt.savefig(f'outputs/df_stats_{prob}/{sbs_name}_{prob}_volplot.png')
    plt.show()
def sbs_stats(aggregate,sbs_id, TF_files, df_DBD_family,
              prob = 0.5,alpha_qvalue = 0.05,log2FC_limit=0.5):
    if not os.path.exists(f"outputs/df_stats_{prob}"):
        os.makedirs(f"outputs/df_stats_{prob}")
        print(f"outputs/df_stats_{prob} folder created!")
    TF_list = []
    for tf in TF_files:
        TF_list.append(tf.split("_")[-1].split(".")[0])
    TF_dict_gain = {tf:0 for tf in TF_list}
    TF_dict_loss = {tf:0 for tf in TF_list}
    sbs_name = sbs_id
    df_sbs = aggregate[aggregate[sbs_name] >= prob]
    TF_dict_gain, TF_dict_loss = pred_TF_counts(df_sbs,TF_dict_gain,TF_dict_loss) # TF counter
    df_merged = pd.DataFrame([TF_dict_gain,TF_dict_loss]).T.reset_index()
    df_merged.columns = ["TF name","#Gain","#Loss"]
    df_merged.sort_values(by="#Gain", ascending=False, inplace=True)
    # PERCENTAGE OF GAIN/LOSS
    df_merged["Percent_gain"] = (df_merged["#Gain"] * 100) / len(aggregate[aggregate[sbs_name] >= prob])
    df_merged["Percent_loss"] = (df_merged["#Loss"] * 100) / len(aggregate[aggregate[sbs_name] >= prob])
    # df_merged["percent_both"] = df_merged["Percent_gain"] +  df_merged["Percent_loss"]
    # FOLD CHANGE Calculations
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
    df_merged["abs(log2FC)"] = abs(df_merged["log2FC"])
    df_merged.sort_values(by="abs(log2FC)",ascending=False,inplace=True)
    n_gain = sum(df_merged["FC"] > 1)
    n_nochange = sum(df_merged["FC"] == 1)
    n_loss = sum(df_merged["FC"] < 1)
    info_sbs = pd.DataFrame([{"n_mutation": len(df_sbs),"n_pos_TF":n_gain,"n_nochange":n_nochange,"n_neg_TF":n_loss}])
    df_stats = fish_exact(df_merged,sbs_name, df_DBD_family, alpha=alpha_qvalue, log2FC_limit=log2FC_limit)
    df_stats.to_csv(f"outputs/df_stats_{prob}/dfstats_{sbs_name}_{prob}.csv",index=False)
    plot_volcano(df_stats,sbs_name,prob,alpha=alpha_qvalue,log2FC_limit=log2FC_limit)
    return info_sbs

SBS_names = df_results.columns[14:].values
prob = 0.75
alpha_qvalue = 0.05
log2FC_limit = 0.25
sbs_info_dict = {}
for sbs in SBS_names:
    sbs_info_dict[sbs] = sbs_stats(df_results,sbs,TF_files, df_DBD_family,prob=prob, alpha_qvalue=alpha_qvalue, log2FC_limit=0.25)
    print(f"{sbs}----------------------------------------: done!")
df_info = pd.concat(sbs_info_dict).reset_index().drop(columns="level_1").rename(columns={"level_0":"Signature"})
df_info.to_csv(f"outputs/df_stats_{prob}/df_sbs_info_{prob}.csv",index=False)

df_sbs2 = pd.read_csv("outputs/dfstats_0.5/dfstats_SBS2.csv")

df_2 = fish_exact(df_sbs2,"SBS2",alpha=0.05,gain_thr=2**(0.5),loss_thr =2**(-0.5))

# def random_zerovalue(row):
#     if row["q_value_fisher"] == 0:
#         expression = f"10e-{random.randint(300, 320)}"
#         result = eval(expression)
#         return result
#     else:
#         return row["q_value_fisher"]

df_p = pd.read_csv("outputs/df_SBS2_tests.csv")

df_sbs_stats = glob.glob("outputs/dfstats/dfstats_SBS*.csv")

df_files = glob.glob("outputs/tests/df*.csv")

for df_file in df_files:
    sbs_name = df_file.split("_")[1].split(".")[0]
    df_p = pd.read_csv(df_file)
    plot_volcano(df_p,sbs_name,prob,log2FC_limit,alpha=0.05)


TF_data = pd.read_csv("TF_model_merged_r2.csv")

TF_data.rename(columns={"Unnamed: 0":"TF_files"},inplace=True)
merged_df = pd.merge(TF_data,df_family,on="TF_Name",how="left")

cols2 = ["Ensembl ID","HGNC symbol","DBD"]
df_family_steven = pd.read_csv("C:/Users/husey/bindFx_storage/TF_family_metainfo/The binding domain classifications,steven rozen.csv",index_col="Unnamed: 0").reset_index(drop="index")
df_family_steven = df_family_steven.loc[:,cols2]
df_family_steven.rename(columns={"Ensembl ID":"Ensembl_Gene_ID","HGNC symbol":"TF_Name","DBD":"DBD"},inplace=True)

TF_data_merge = pd.merge(TF_data,df_family_steven,on="TF_Name",how="left")
TF_data_merge.to_csv("mergedv1.csv",index=False)

df_DBD = pd.read_csv("TF_model_family.csv")


def heat_pie_chart(data_array,
                           row_labels, # normally rows TF DBD families
                           column_labels, # normally columns Signatures
                           prob,
                           filename="SBS_TF_heat_piechart_v1",
                           plot_legend=True,
                           number_of_columns_in_legend = 3,
                           heatmap_rows_signatures_columns_dna_elements = True,
                           dpi = 100,
                           squaresize = 200,
                           fontsize = 60,
                           wedges_fontsize = 30,
                           labeldistance = 0.20 ):

    if len(column_labels) <= 3:
        figwidth = 2 * len(column_labels) * squaresize / float(dpi)
    else:
        figwidth = len(column_labels) * squaresize / float(dpi)

    figheight = len(row_labels) * squaresize / float(dpi)
    fig, ax = plt.subplots(1,figsize=(figwidth, figheight), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_aspect(1.0)
    ax.set_facecolor('white')
    for row_index, row in enumerate(row_labels):
        for column_index, column in enumerate(column_labels):
            # Values is a tuple of 3 integer
            values = eval(data_array[row_index, column_index])
            labels = [str(value) for value in values]
            gof,lof,no = 'indianred', 'cornflowerblue', 'silver'
            # gof,lof,no = 'tab:red', 'tab:blue', 'silver'
            colors = [gof,lof,no]
            # Removes 0s
            mask = [True if value > 0 else False for value in values]
            values = (np.array(values)[mask]).tolist()
            labels = (np.array(labels)[mask]).tolist()
            colors = (np.array(colors)[mask]).tolist()

            wedges, text = ax.pie(values, labels = labels, labeldistance = labeldistance, colors = colors, textprops = {'fontsize': 30}) # legacy labeldistance = 0.20 fontsize 30
            radius = 0.45
            for w in wedges:
                # We want to see the data_array[0,0] at left most top instead of left most bottom
                w.set_center((column_index, len(row_labels)-row_index-1)) # legacy
                w.set_radius(radius)
                # w.set_edgecolor('white')
            for t in text:
                x, y = t.get_position()
                # We want to see the data_array[0,0] at left most top instead of left most bottom
                t.set_position((x + column_index, y + len(row_labels)-row_index-1)) # legacy

    legend_elements = [
        Line2D([0], [0], marker='o', color='white', label='GOF', markerfacecolor=gof, markersize=60),
        Line2D([0], [0], marker='o', color='white', label='LOF', markerfacecolor=lof, markersize=60),
        Line2D([0], [0], marker='o', color='white', label='No effect', markerfacecolor=no, markersize=60)]

    if plot_legend:
        if heatmap_rows_signatures_columns_dna_elements:
            ax.legend(handles=legend_elements, ncol = number_of_columns_in_legend, loc="upper right", bbox_to_anchor=(1, 0), fontsize=50) # one row
        else:
            # rows DNA elements, columns signatures
            # when there are a few signatures, it would be good to have legends in one column
            ax.legend(handles=legend_elements, ncol = 1, loc="upper right", bbox_to_anchor=(1, 0), fontsize=50) # one row

    # We want to show all ticks...
    ax.set_xticks(np.arange(data_array.shape[1]))
    ax.set_yticks(np.arange(data_array.shape[0]))

    # Remove string from columns
    # column_labels = ['Open Chromatin' if 'ATAC' in label else label for label in column_labels]
    # row_labels = ['Open Chromatin' if 'ATAC' in label else label for label in row_labels]
    # Change Dinucleotides to Doublets
    # row_labels = [row.replace("Dinucleotides", "Doublets") if 'Dinucleotides' in row else row for row in row_labels]
    # column_labels = [column.replace("Dinucleotides", "Doublets") if 'Dinucleotides' in column else column for column in column_labels]

    # Tick labels are set since ticks are set above
    ax.set_xticklabels(column_labels, fontsize=fontsize)
    # Reverse the row labels so that left most top is the first one
    # Otherwise left most bottom is the first one
    # Do not use row_labels.reverse() since it reverse in place and return nothing
    # row_labels[::-1] returns the reversed list, do not reverse the input list
    ax.set_yticklabels(row_labels[::-1], fontsize=fontsize)

    # X axis labels at top
    ax.tick_params(left=False, top=False, bottom=False, right=False, labelbottom=False, labeltop=True, pad=5)
    plt.setp(ax.get_xticklabels(), rotation=55, ha="left", rotation_mode="anchor")

    # We want to show all minor ticks...
    # ax.set_xticks(np.arange(data_array.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data_array.shape[0] + 1) - .5, minor=True)

    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xlim(xmin = -0.5)
    if len(row_labels) > 1:
        ax.set_ylim(ymin = -0.5)
    elif len(row_labels) == 1:
        ax.set_ylim(ymin = -0.5,ymax = 0.5)
    fig.savefig(f"outputs/{filename}.png", dpi=dpi, bbox_inches="tight")
    plt.cla()
    plt.close(fig)

def sort_key(x):
    # Extracting the numeric part and converting it to integer
    return int(x[3:])

def family_based_pie_chart(df_sbs_stats,filename,prob=0.5):
    SBS_TF_family_dict = dict()
    for sbs in df_sbs_stats:
        SBS_name = sbs.split("_")[3]
        df = pd.read_csv(sbs)
        # plot_volcano(df,SBS_name,prob=prob,alpha=0.05,log2FC_limit=0.5)
        TF_family_dict = dict(zip(df["DBD_family"].unique(),[[0,0,0]]*len(df["DBD_family"].unique())))
        for x in TF_family_dict.keys():
            n_gof = sum(df[df["DBD_family"] ==x ]["TF_binding"] == "GOF")
            n_lof = sum(df[df["DBD_family"] ==x ]["TF_binding"] == "LOF")
            n_notr = sum(df[df["DBD_family"] ==x ]["TF_binding"] == "Neutral")
            TF_family_dict[x] = [n_gof,n_lof,n_notr]
        SBS_TF_family_dict[SBS_name] =  TF_family_dict
    df_SBSvsFamily = pd.DataFrame(SBS_TF_family_dict)
    sorted_columns = pd.Series(df_SBSvsFamily.columns).sort_values(key=lambda x: x.map(sort_key)).values
    sorted_rows = np.array(sorted(df_SBSvsFamily.index.values, key=lambda x: x.lower()))
    df_SBSvsFamily = df_SBSvsFamily.loc[sorted_rows,sorted_columns]
    df_SBSvsFamily.to_csv(f"outputs/df_stats_{prob}/df_SBSvsTF_DBD_Family_{prob}.csv")
    # Filtering rows
    rows_to_drop = df_SBSvsFamily.apply(lambda row: all([cell[0] == 0 and cell[1] == 0 for cell in row]), axis=1)
    df_SBSvsFamily = df_SBSvsFamily[~rows_to_drop].drop(index="Unknown")
    # Filtering columns
    cols_to_drop = df_SBSvsFamily.apply(lambda col: all([cell[0] == 0 and cell[1] == 0 for cell in col]), axis=0)
    df_SBSvsFamily = df_SBSvsFamily.loc[:, ~cols_to_drop]
    data_array = df_SBSvsFamily.values
    row_labels = df_SBSvsFamily.index.values
    column_labels = df_SBSvsFamily.columns.values

    heat_pie_chart(data_array,row_labels,column_labels,prob,filename=filename)

df_sbs_stats = glob.glob("outputs/df_stats_0.75_0.25/dfstats_SBS*")


sum_ln = {}
for x in df_sbs_stats:
    df = pd.read_csv(x)
    sum_ln["lof"] = sum(df["TF_binding"] == "LOF")
    sum_ln["gof"] = sum(df["TF_binding"] == "GOF")
    sum_ln["neutral"] = sum(df["TF_binding"] == "Neutral")

family_based_pie_chart(df_sbs_stats,"heatmap_piechart_0.5",prob=0.5)

df_family = pd.read_csv("outputs/df_stats_0.75_0.25/df_SBSvsTF_DBD_Family_0.75_nonzero.csv", index_col="Unnamed: 0")

filename = "family_forkhead"
prob = 0.75

df_SBSvsFamily = pd.DataFrame(df_family.loc["Forkhead"]).T
