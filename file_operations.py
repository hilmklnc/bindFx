import pandas as pd
import random as rd
import os
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
# FILE OPERATION

with open("CTCF/    ") as f:  # fasta file to extract sequences
    fasta = f.readlines()
locs = [y.split()[0].replace(">","").replace("(+)","") for x,y in enumerate(fasta) if x%2 == 0]
seqs = [y.rstrip().upper().replace("N","") for x,y in enumerate(fasta) if x%2 != 0]

bed_data = pd.read_csv("CTCF/ENCFF349RNE.bed",  sep="\t",header=None) # bed narrowpeak file to extract scores
bed_data.columns = ["chrom","chromStart","chromEnd","name","score","strand","signalValue","pValue","qValue","peak"]
bed_data.head()

chipseq_train = pd.DataFrame({"region": locs, "sequence": seqs,"score":bed_data["signalValue"],"peak": bed_data["peak"]})

# ensembl_check = "GCAGTCATGGGTCCTTAATGAGAGGGATACATTCTGAGAAATGTGTCATTAGGCAGTGTCATTGTGTAAACATTATAGAGTGTGTTTACACAAACTTAGGTGGCATAGTCCCCTACTCTCCTAAGCTATATGGCATAGCCTATTTTCTTCTGGGCTACAAACCTGTATAGCATGTTACTGTACTGAATACTCTAGGCAATTATAACATAATAGTAAGTGTTTGTGTATCTAAACAAGAAAAATAGTACAGTAAGAATATTGTATAAAAGCTCCATTATAATCTCATGGGCCCACTGGAGTTTCACCATGGTTGACTGAAAGGTCTTTATGTGGCGCATGACTGTATTTTGTAATTCATAAATTTCACCTTATCGGACAGTGCCTCCTCTCTGTGCTTTGCGTTAGAGTAGATATGCTTGCTTATCAGGGGTTGGCAGAGTTTTTCTGTGAAGGGCTAGATAGCAGATATTTTGGACTTGTGGGCCACATGGACTGTGGAGGACTCTGGCAACTATTCAGCTCTGCCATTGTGGCATGAAACCAGCCATAGATAATATATAAACAAGTGAGTATGGCTTCATTCCAATACAACTTTTATTTATACAAACAGGCAGCAGGCTAAATATGACCTGAAAGTCATATTTTGCTGGCCCCTCCTCTATACCATTATGGAAAGATATAGAAATCCAGTGTGCTTCACTCTTGCCACTAAAATCTCCTTACTGTGCTCTCAGTGATAAGTGTTAAAACATGAGTTGAAAAACTTGAACTTCTCAAAATCTGATTAATATGCATGGGCATAATCTATTTCTTTTTTCTTTGTTTTTTTTTTTTTTTTTTTTTTGAGGTAGAGTCTTCCTCTGTTGTCCAAGCTGGAGTGTAGTGGCCCGATCTCGGGTTCAAGTGATTCTCCTGCC"
# ensembl_check == chipseq_train["sequence"][0]


# Prepare Complement 60 bps file
positive_bed = pd.read_csv("/home/hilmi/qbic/GATA1/GATA1_bed/ENCFF853VZF.bed",  sep="\t",header=None) # bed narrowpeak file to extract scores
positive_bed.columns = ["chrom","chromStart","chromEnd","name","score","strand","signalValue","pValue","qValue","peak"]
positive_bed.head()

non_bind = pd.read_csv("complemented_39bed.bed", header=None, sep="\t") # 437.322 rows - chr/start/end
#non_bind.loc[non_bind[1]==0 ,1] = 1
#sum(non_bind[1]== 0)
#non_bind.to_csv("non_binding_1s.txt",header=False, index=False, sep="\t")

non_bind = non_bind[(non_bind[1]+180) < non_bind[2]].reset_index(drop="index")
sum(positive_bed[0] == f"chr{1}")
sum(non_bind[0] == "chr1")
sum(non_bind[1]+61>non_bind[2])

random_bed = pd.DataFrame(columns=["chrom","start","end"])

def generator_60(x):
    a = rd.randrange((x[1].values[0] + 61), x[2].values[0])  # end point
    b = a - 60  # start point
    return a,b

def new_df(df):
    for i in range(len(df)):
        df[2][i] = rd.randrange((df[1][i] + 61), df[2][i])  # end point
        df[1][i] = df[2][i] - 60  # start point

new_bed = new_df(non_bind)
new_bed = non_bind
bed_data = positive_bed
sum(bed_data["chrom"] == f"chr{1}")
# sum(chipseq_negatives["region"] == f"chr{1}")

# for i in range(1,23): # i = chr type
#     for j in range(sum(bed_data["chrom"] == f"chr{i}")):
#         sample_chr = non_bind[non_bind[0] == f"chr{i}"].sample()
#         a,b = generator_60(sample_chr)
#         new_row = pd.DataFrame([{"chrom":f"chr{i}","start":b,"end":a}])
#         random_bed = pd.concat([random_bed,new_row])
# for j in range(sum(bed_data["chrom"] == "chrX")):
#     sample_chr = non_bind[non_bind[0] == "chrX"].sample()
#     a, b = generator_60(sample_chr)
#     random_bed = random_bed.append({"chrom": "chrX", "start":b,  "end":a}, ignore_index=True)
# for j in range(sum(bed_data["chrom"] == "chrY")):
#     sample_chr = non_bind[non_bind[0] == "chrY"].sample()
#     a, b = generator_60(sample_chr)
#     random_bed = random_bed.append({"chrom": "chrY", "start":b,  "end":a}, ignore_index=True)
#
# new_bed = random_bed

# (new_bed["end"]-new_bed["start"]).describe()
(new_bed[2]-new_bed[1]).describe()

new_bed.to_csv("complemented_60_39files_2.bed",header=False, index=False, sep="\t")

# Fasta Operation
#------------------------------------------------------------------------------------
# Choose 60 bp DNA strings after MEME-tool fasta conversion ( some 60 bps contain NA)
with open("complemented_60_39files_2_fasta.txt") as f:  # fasta file to extract sequences
    fasta = f.readlines()
locs = [y.split()[0].replace(">","").replace("(+)","") for x,y in enumerate(fasta) if x%2 == 0]
seqs = [y.rstrip().upper().replace("N","") for x,y in enumerate(fasta) if x%2 != 0]

chipseq_negatives = pd.DataFrame({"region": locs, "sequence": seqs,"score":0,})

sum(chipseq_negatives["sequence"].apply(len) != 60)
# getting only length of 60 bps
chipseq_negatives = chipseq_negatives[chipseq_negatives["sequence"].apply(len) == 60].reset_index(drop="index")
all(chipseq_negatives["sequence"].apply(len) == 60)


# Merging Negative and positive sets
#----------------------------------------------------------------------------------
chipseq_positive = pd.read_csv("/Users/husey/qbic/GATA1/GATA1chip_pbmformat/ChIPseq_ENCFF853VZF_GATA1.txt", header=None, sep="\t")
chipseq_positive1 = pd.read_csv("/Users/husey/qbic/GATA1/GATA1chip_pbmformat/ChIPseq_ENCFF657CTC_GATA1.txt", header=None, sep="\t")
chipseq_positive2 = pd.read_csv("/home/hilmi/qbic/GATA1/GATA1chip_pbmformat/ChIPseq_ENCFF509ZLE_GATA1.txt", header=None, sep="\t")
chipseq_positive_combined = pd.concat([chipseq_positive, chipseq_positive1])
chipseq_positive_combined.reset_index(drop="index",inplace=True)
chipseq_positive_combined.sort_values(by=[0],ascending=False,inplace=True)
chipseq_positive_combined.to_csv("VZF_CTC_combined.txt",header=False, index = False, sep="\t")
chipseq_negative = pd.read_csv("negative_trainset_60_forVZF2",header=None,sep="\t")
chipseq_negative = chipseq_negatives.sample(len(chipseq_positive_combined)).reset_index(drop="index")

# chipseq_negative = chipseq_negative.sample(100000).reset_index(drop="index")
sum(chipseq_negative["sequence"].str.contains("CTTATC")) + sum(chipseq_negative["sequence"].str.contains("GATAAG"))

chipseq_positive.rename(columns={0:"score",1:"sequence"},inplace=True)

# chipseq_positive = pd.concat([chipseq_positive,chipseq_positive1,chipseq_positive2]).sort_values(by=0,ascending=False)
# chipseq_positive.reset_index(drop="index",inplace=True)

# chipseq_positive = chipseq_positive[chipseq_positive[0] > 100]
# chipseq_positive[0] = chipseq_positive[0]*100
chipseq_negative = chipseq_negative.sample(len(chipseq_positive)).reset_index(drop="index")
chipseq_negative[1].apply(len).describe()
chipseq_positive[1].apply(len).describe()

# chipseq_negative = chipseq_negative.sample(100000).reset_index(drop="index")
sum(chipseq_negatives["sequence"].str.contains("CTTATC")) + sum(chipseq_negatives["sequence"].str.contains("GATAAG"))
chipseq_negative = pd.concat([chipseq_negative["score"],chipseq_negative["sequence"]],axis=1)
chipseq_negative.to_csv("negativeset_60_vzf_2.txt",header=False, index=False, sep="\t")
chipseq_negative.columns = [0,1]

# chipseq_positive["score"] = logshift(chipseq_positive["score"])
# third_quartile = np.percentile(chipseq_positive["score"], 75)
# chipseq_positive = chipseq_positive[chipseq_positive["score"] > third_quartile]
# bed_data = positive_bed.iloc[:len(chipseq_positive),:]
#----------------------------------------------------------------------------------

# Loading negative set
chipseq_negative = pd.read_csv("/home/hilmi/qbic/March_2_trials/negativeset_60_all.txt", header=None, sep="\t")
# Equally distributed data:
chipseq_negative = chipseq_negatives.sample(len(chipseq_positive)).reset_index(drop="index")

chipseq_negative[1].apply(len).describe()
chipseq_positive[1].apply(len).describe()

# chipseq_negative = chipseq_negative.sample(100000).reset_index(drop="index")
sum(chipseq_negative["sequence"].str.contains("CTTATC")) + sum(chipseq_negative["sequence"].str.contains("GATAAG"))
#------------------------------------------------------------------

# random sample after 60_fasta
chipseq_negatives["region"]= chipseq_negatives["region"].str.rsplit(":", expand=True)[0]
bed_data = positive_bed
non_bind = chipseq_negatives
new_equal_negatives = pd.DataFrame()
for i in range(1,23): # i = chr type
        sample_row = non_bind[non_bind["region"] == f"chr{i}"].sample(sum(bed_data["chrom"] == f"chr{i}"))
        new_equal_negatives = pd.concat([new_equal_negatives,sample_row])
sample_row = non_bind[non_bind["region"] == "chrX"].sample(sum(bed_data["chrom"] == "chrX"))
new_equal_negatives = pd.concat([new_equal_negatives, sample_row])
sample_row = non_bind[non_bind["region"] == "chrY"].sample(sum(bed_data["chrom"] == "chrY"))
new_equal_negatives = pd.concat([new_equal_negatives, sample_row])

chipseq_negative = pd.concat([new_equal_negatives["score"],new_equal_negatives["sequence"]],axis=1,).reset_index(drop="index")
chipseq_negative.to_csv("negative_trainset_unionGATA1_half",header=False, index = False, sep="\t")
chipseq_negative.columns = [0,1]
trainset = pd.concat([chipseq_positive_combined, chipseq_negative], ignore_index=True)
trainset.to_csv("negatived_VZF_CTC",header=False, index = False, sep="\t")


# Check their distribution:
negative_bed = pd.read_csv("/home/hilmi/qbic/negative_trainset_60_unionGATA1", header=None,sep="\t")
chipseq_negative = negative_bed
def chr_distribution(bed1,bed2): # bed1: negative , bed2: positive
    for i in range(1,23):
        print(f"chr{i} equality:",sum(bed1["region"] == f"chr{i}") == sum(bed2["chrom"] == f"chr{i}"))  # double check eden function, boxplot
    print("chrX equality:",sum(bed1["region"]=="chrX")  == sum(positive_bed["chrom"]=="chrX")) # double check eden function, boxplot
    print("chrY equality:",sum(bed1["region"]=="chrY")  == sum(positive_bed["chrom"]=="chrY")) # double check eden function, boxplot

chr_distribution(new_equal_negatives,bed_data)

#-------------------------------------------------------------

# Modifying the negativeset
def drop_motif(chipseq_negative):
    cttatc = chipseq_negative[chipseq_negative[1].str.contains("CTTATC")].index
    chipseq_negative = chipseq_negative.drop(cttatc).reset_index(drop="index")
    gataag = chipseq_negative[chipseq_negative[1].str.contains("GATAAG")].index
    chipseq_negative = chipseq_negative.drop(gataag).reset_index(drop="index")
    # gataac = chipseq_negative[chipseq_negative[1].str.contains("GATAAC")].index
    # chipseq_negative = chipseq_negative.drop(gataac).reset_index(drop="index")
    # motif = chipseq_negative[chipseq_negative[1].str.contains("GTTATC")].index
    # chipseq_negative = chipseq_negative.drop(motif).reset_index(drop="index")
    # motif = chipseq_negative[chipseq_negative[1].str.contains("CTGATA")].index
    # chipseq_negative = chipseq_negative.drop(motif).reset_index(drop="index")
    # motif = chipseq_negative[chipseq_negative[1].str.contains("TATCAG")].index
    # chipseq_negative = chipseq_negative.drop(motif).reset_index(drop="index")
    # # motif = chipseq_negative[chipseq_negative[1].str.contains("GATAAA")].index
    # chipseq_negative = chipseq_negative.drop(motif).reset_index(drop="index")
    # motif = chipseq_negative[chipseq_negative[1].str.contains("TTTATC")].index
    # chipseq_negative = chipseq_negative.drop(motif).reset_index(drop="index")
    # motif = chipseq_negative[chipseq_negative[1].str.contains("AGATAA")].index
    # chipseq_negative = chipseq_negative.drop(motif).reset_index(drop="index")
    # motif = chipseq_negative[chipseq_negative[1].str.contains("TTATTC")].index
    # chipseq_negative = chipseq_negative.drop(motif).reset_index(drop="index")
    return chipseq_negative
chipseq_negative = drop_motif(chipseq_negative) # Modified negative set
trainset = pd.concat([chipseq_positive, chipseq_negative], ignore_index=True)
trainset.to_csv("trainset_negatived_vzf",header=False, index = False, sep="\t")



# -------------------------------------------------#
# OTHER TFs Pipeline
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

def bedtotrainset(filefasta,filebed,offset,outputname):
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


# Gathering Human PBM datasets from CISBP databases
os.makedirs("hs_pbm_data2")
all_pbm = os.listdir("/Users/husey/Desktop/CISBP_SignalIntensities")

data_human_pbm = pd.read_csv("hs_TF_PBM_info.csv")
data_human_pbm["TF_Name"].nunique()
motif_id = data_human_pbm["Motif_ID"].values

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


import requests
import os
import pandas as pd

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




df = pd.read_csv("outputs/encode_all_results_CV.csv")

sum(df["r2"] < -1)
sum(df["r2"] < -0.5)
sum(df["r2"] < 0)
sum(df["r2"] >= 0.1)
sum(df["r2"] >= 0.15)
sum(df["r2"] >= 0.2)
sum(df["r2"] >= 0.22)
sum(df["r2"] >= 0.25)
sum(df["r2"] >= 0.3)
sum(df["r2"] >= 0.4)

sns.histplot(data = df, x="r2",color= sns.color_palette('Set3')[0])
plt.xlim([-1,1])
plt.title("R2 metric with 10-fold CV")
plt.show()

ranges = [(0, 2080), (2080, 5000), (5000, 10000), (10000 , 20000), (20000, 100000),(100000,df["n_peaks"].max())]

grouped_counts = []
for lower, upper in ranges:
    count = df[(df['n_peaks'] >= lower) & (df['n_peaks'] < upper)].shape[0]
    grouped_counts.append(count)

plt.figure(figsize=(10,8))
labels = [f'{lower}-{upper} peaks' for lower, upper in ranges]
plt.pie(grouped_counts, labels=labels, autopct="%1.1f%%",wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'},
        labeldistance=1.18,startangle=40,colors=sns.color_palette('Set2'))
plt.setp(plt.gca().get_children()[1], ha='center')  # This line centers the percent labels inside each pie slice
plt.legend([f'{label}: {count}' for label, count in zip(labels, grouped_counts)], loc="lower right",fontsize=8)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# Select the highest r2 performance between each others
perf_df = pd.read_csv("outputs/encode_all_results.csv",index_col="Unnamed: 0")
perf_df
TF_names = perf_df["TF_Name"].unique()
len(TF_names)
targets = []
for tf in TF_names:
    ind = perf_df[perf_df["TF_Name"] == tf]["r2"].idxmax()
    targets.append(ind)
len(targets) # 254 unique TF datasets

perf_df.loc[targets].to_csv("outputs/encode_unique_results.csv")

pd.Series(targets).to_csv("outputs/encode_targets.csv")