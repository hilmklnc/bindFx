import pandas as pd
import random as rd
import os
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import requests


# From fasta to trainset format sample
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

def bedtotrainset(filefasta,filebed,offset,outputname): # input fasta
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
    chipseq_train["peak_seq"] = chipseq_train.apply(lambda x: peak_offset(x["sequence"], x["peak"], offset), axis=1)
    pbm_format = pd.DataFrame({0: chipseq_train["score"], 1: chipseq_train["peak_seq"]})
    pbm_format.sort_values(by=0,ascending=False,inplace=True)
    pbm_format.to_csv(outputname, header=None, index=False, sep="\t")
    return pbm_format

bedtotrainset("21breast_VCF/sample_fasta/PD3904a_fasta","21breast_VCF/sample_fasta/PD3904a_bed",
              30,"ChIPseq_ENCFF682EOV_CTCF.txt")

ctcf = pd.read_csv("ChIPseq_ENCFF349RNE_CTCF.txt", header=None, sep="\t")

#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################

# Gathering Human PBM datasets from CISBP databases
os.makedirs("combined_trainsets")
folder_path = "outputs/PBM_trainsets"
all_pbm = os.listdir(folder_path)

data_human_pbm = pd.read_csv("hs_TF_PBM_info.csv")
data_human_pbm["TF_Name"].nunique()
motif_id = data_human_pbm["Motif_ID"].values
#
#Extracting PBM data from CISBP according to motif_ID
for filename in all_pbm:
    for id in motif_id:
        if id in filename:
            match_path = os.path.join("/Users/husey/Desktop/CISBP_SignalIntensities",filename)
            TF_name = data_human_pbm[data_human_pbm["Motif_ID"] == id ]["TF_Name"].values[0]
            new_filename = filename.split(".txt")[0] + "_" + TF_name + ".txt"
            copy_path = os.path.join("hs_pbm_data2",new_filename)
            shutil.copy(match_path,copy_path)
            print(f"{TF_name} Copied {filename} and {id} to hs_pbm_data folder")
            print("next-->")
# Rename the PBM files

def rename_files_in_folder(folder_path, new_name):
    for filename in os.listdir(folder_path):
        # construct old file path
        old_file = os.path.join(folder_path, filename)
        # construct new file path
        new_filename = new_name + "_" + filename
        new_file = os.path.join(folder_path, new_filename)
        # rename the file
        os.rename(old_file, new_file)
        print(f"{new_filename} Copied to hs_pbm_data folder!")
        print("next-->")
# usage
folder_path = "/Users/husey/bindFx_storage/data/PBM_trainsets"
new_name = "PBM"
rename_files_in_folder(folder_path, new_name)

# For LAI20A file:
# new_filename = ("_".join([new_name, filename.split("_")[-1].split(".")[0], filename.split("_")[-2], filename.split("_")[0].upper()])
#                + "." + filename.split("_")[-1].split(".")[-1])

#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################

# DOWNLOAD ENCODE URL FILES
# List of URLs
urls = pd.read_csv("encode_files.txt")
urls = urls[urls.columns[0]]

type(urls)
# Create a folder to store downloaded files if it doesn't exist
folder_name = "downloaded_files"
os.makedirs(folder_name, exist_ok=True)
urls[0]
# Function to download files
def download_file(url):
    filename = url.split("/")[-1]  # Extract filename from URL
    file_path = os.path.join(folder_name, filename)  # Path to save the file

    # Download the file
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    print(f"Downloaded {filename}")


# Download files from URLs
for url in urls:
    download_file(url)


#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################

# Select the highest r2 performance between each others
perf_df = pd.read_csv("outputs/PBM_all_results_10foldCV.csv",index_col="Unnamed: 0")
perf_df_chip = pd.read_csv("outputs/encode_all_results_10foldCV_run1.csv",index_col="Unnamed: 0")
perf_df = pd.read_csv("outputs/encode_all_results_run1.csv",index_col="Unnamed: 0")

len(perf_df_chip["TF_Name"])
len(perf_df_chip["TF_Name"].unique())
TF_names = perf_df["TF_Name"].unique()
TF_names = perf_df_chip["TF_Name"].unique()

def unique_TF(df,TF_names):
    targets = []
    for tf in TF_names:
        ind = df[df["TF_Name"] == tf]["r2"].idxmax()
        targets.append(ind)
    return targets
targets = unique_TF(perf_df,TF_names)
len(targets) # == len(unique TF names)

# CHIP
perf_df_chip.loc[targets].to_csv("outputs/encode_unique_maxr2_10foldCV.csv")
unique_perf_df_chip = perf_df.loc[targets]
new_unique_filtered_chip = unique_perf_df_chip[unique_perf_df_chip["r2"] >= 0.2]
new_unique_filtered_chip.to_csv("outputs/encode_filtered_results.csv")

# PBM
perf_df.loc[targets].to_csv("outputs/pbm_unique_maxr2_10foldCV.csv")
unique_perf_df = perf_df.loc[targets]
new_unique_filtered_pbm = unique_perf_df[unique_perf_df["r2"] >= 0.1]
new_unique_filtered_pbm.to_csv("outputs/pbm_filtered_results.csv")

# PBM and CHIP
merged_filtered_df = pd.concat([new_unique_filtered,new_unique_filtered_chip])
len(merged_filtered_df["TF_Name"])
len(merged_filtered_df["TF_Name"].unique())
TF_names = merged_filtered_df["TF_Name"].unique()
all_targets = unique_TF(merged_filtered_df,TF_names)
merged_unique_df = merged_filtered_df.loc[all_targets]
merged_unique_df.to_csv("outputs/merged_pbm_chip_final.csv")


merged_unique_df.index[:249]

merged_unique_df.index[249:]
# list1 = target_2
# list2 = target_1
# difference_set = set(list2) ^ set(list1)
# common_elements = set(list2) & set(list1)
# len(difference_set)
# len(common_elements)
#
# common_elements = set(pbm_uniq) & set(chip_uniq)
# difference_set = set(pbm_uniq) ^ set(chip_uniq)
# len(common_elements)
# len(difference_set)

# COPY AND PASTE CHIP/PBM FILES ACCORDING TO OUR TARGET FILENAMES
all_files = list(merged_unique_df.index)
os.makedirs("all_trainsets")
for i,filename in enumerate(all_files):
    if "PBM" in filename:
        print("PBM file found!\n")
        match_path = os.path.join("outputs/PBM_trainsets",filename)
        copy_path = os.path.join("all_trainsets",filename)
        shutil.copy(match_path,copy_path)
        print(f"{filename} Copied!")
    elif "ChIP" in filename:
        print("ChIP file found!\n")
        match_path = os.path.join("outputs/encode_trainsets",filename)
        copy_path = os.path.join("all_trainsets",filename)
        shutil.copy(match_path,copy_path)
        print(f"{filename} Copied!")
    else:
        print("There is a problem!!!!!!!")
    print("\nnext-->",i+1)

check_trainsets = glob.glob("all_trainsets/*.txt")
all_files_2 = []
for x in check_trainsets:
    all_files_2.append(os.path.split(x)[1])
print(sorted(all_files_2) == sorted(all_files))

#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################

# TF DATASET REPORT AND VISUALIZATION
df = pd.read_csv("outputs/encode_all_results_10foldCV_run1.csv")

sns.histplot(data = df, x="r2",color= sns.color_palette('Set3')[0])
plt.xlim([-1,1])
plt.title("ChIP+PBM data with 10-fold CV")
plt.show()

ranges = [(2080, 5000), (5000, 10000), (10000 , 20000), (20000, df["n_peaks"].max()+1)]

grouped_counts = []
for lower, upper in ranges:
    count = df[(df['n_peaks'] >= lower) & (df['n_peaks'] < upper)].shape[0]
    grouped_counts.append(count)

plt.figure(figsize=(10,8))
labels = [f'{lower}-{upper} peaks' for lower, upper in ranges]
plt.pie(grouped_counts, autopct="%1.1f%%",wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'},
        labeldistance=1.18,startangle=30,colors=sns.color_palette('Set3'))
plt.setp(plt.gca().get_children()[1], ha='center')  # This line centers the percent labels inside each pie slice
plt.legend([f'{label}: {count}' for label, count in zip(labels, grouped_counts)], loc="lower right",fontsize=9.5)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("ChIP+PBM data Peak Distribution")
plt.show()


