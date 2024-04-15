import glob
import numpy as np
# import pyfastx
import biocoder
import pandas as pd
from pyfaidx import Fasta
#--------------------------------------------------------------------------#

# vep format
def altered_seq(df): # fetching -30/+30 sequences from the center mutation point
    center = 30
    dna_sequences = df["sequence"]
    allele_ref, allele_alt = df["ref"], df["alt"]

    if len(allele_ref) <= len(allele_alt):
        if allele_ref == "-":  #insertion (-/NNNNN)
            altered_sequence = dna_sequences[:(center+1)] + allele_alt + dna_sequences[(center+1) :]
        elif len(allele_ref) == len(allele_alt): # balanced point mutation or doublets+
            if allele_ref == "-": # point insertion (-/N)
                altered_sequence = dna_sequences[:(center+1)] + allele_alt + dna_sequences[(center + 1):]
            elif allele_alt == "-": # point deletion (N/-)
                altered_sequence = dna_sequences[:center] + dna_sequences[center + len(allele_ref):]
            else: # (N/N)
                altered_sequence = dna_sequences[:center] + allele_alt + dna_sequences[(center + 1):]
        else: # unbalanced insertion (N/NNNN)
            altered_sequence = dna_sequences[:center] + allele_alt + dna_sequences[(center + 1 + len(allele_ref)):]
    else:  # Deletion
        if allele_alt == "-": # deletion (NNN/-)
            altered_sequence = dna_sequences[:center] + dna_sequences[center + len(allele_ref):]
        else: # unbalanced deletion # (NNNN/N)
            altered_sequence = dna_sequences[:center] + allele_alt + dna_sequences[(center + 1 + len(allele_ref)):]
    return altered_sequence


def vep_to_bed(vep_file):
    # VEP_raw data retrieved from Sana
    vep_data = pd.read_csv(vep_file)
    vep_data.drop_duplicates(inplace=True) # duplications available
    vep_data = vep_data.reset_index(drop=True)
    vep_data["chr"] = "chr" + vep_data["chr"] # adding chr format for MEME tool
    vep_data["start"] = vep_data["start"]-1 # bed format

    ind = vep_data[vep_data["chr"] == "chrUn_KI270742v1"].index[0] # discard alternative chr variant
    vep_data = vep_data.drop(index=ind).reset_index(drop=True)

    bed_format = pd.concat([vep_data["chr"], (vep_data["start"]-30),(vep_data["start"]+30)],axis=1) # getting -30+ sequences
    bed_format.columns = ["chr","start","end"]
    bed_format.to_csv("vep_to_bed.txt",header=False, index=False,sep="\t")
    print("VEP file is converted to bed format!")
    return  bed_format

def bed_to_seq(vep_file,bed_file,genome,id):
    ref_genome = pyfastx.Fasta(f"genome_assembly/{genome}.fa.gz")
    with open(bed_file, "r") as bed_data, open(f"{id}_fasta.txt", "w") as output_file:
        for line in bed_data:
            chrom, start, end = line.strip().split("\t")[:3]
            start = int(start)
            end = int(end)
            sequence = ref_genome[chrom][start:end]
            output_file.write(f">{chrom}:{start}-{end}\n")
            output_file.write(str(sequence) + "\n")
    # Fetching Sequences in 60 bps
    with open(f"{id}_fasta.txt") as f:  # fasta file to extract sequences
        fasta = f.readlines()
    locs = [y.split()[0].replace(">","").replace("(+)","") for x,y in enumerate(fasta) if x%2 == 0]
    seqs = [y.rstrip().upper().replace("N","") for x,y in enumerate(fasta) if x%2 != 0]
    vep_60seq = pd.DataFrame({"region": locs, "sequence": seqs})
    vep_data = pd.read_csv(vep_file)
    vep_data["sequence"] = vep_60seq["sequence"]
    vep_data["altered_seq"] = vep_data.apply(altered_seq, axis=1)
    vep_data.to_csv("VEP_seq.csv", index=False)

vep_to_bed("VEP_sana/dataset_VEP.csv") # bed_file
bed_to_seq("VEP_sana/dataset_VEP.csv",bed_file,"hg19","vep_noncoding")
vep_seq = pd.read_csv("VEP_seq.csv")


#----------------------------------------------------------------------------

# VCF FILE OPERATIONS

def altered_seq(df): # Creating VCF mutation
    center = 5
    dna_sequences = df["sequence"]
    allele_ref, allele_alt = df[3], df[4]

    if len(allele_ref) <= len(allele_alt):
        if allele_ref == "-":  #insertion (-/NNNNN)
            altered_sequence = dna_sequences[:(center+1)] + allele_alt + dna_sequences[(center+1) :]
        elif len(allele_ref) == len(allele_alt): # balanced point mutation or doublets+
            if allele_ref == "-": # point insertion (-/N)
                altered_sequence = dna_sequences[:(center+1)] + allele_alt + dna_sequences[(center + 1):]
            elif allele_alt == "-": # point deletion (N/-)
                altered_sequence = dna_sequences[:center] + dna_sequences[center + len(allele_ref):]
            else: # (N/N)
                altered_sequence = dna_sequences[:center] + allele_alt + dna_sequences[(center + len(allele_alt)):]
        else: # unbalanced insertion (N/NNNN)
            altered_sequence = dna_sequences[:center] + allele_alt + dna_sequences[(center + 1 + len(allele_ref)):]
    else:  # Deletion
        if allele_alt == "-": # deletion (NNN/-)
            altered_sequence = dna_sequences[:center] + dna_sequences[center + len(allele_ref):]
        else: # unbalanced deletion # (NNNN/N)
            altered_sequence = dna_sequences[:center] + allele_alt + dna_sequences[(center + 1 + len(allele_ref)):]
    return altered_sequence
def single_vcf_to_bed(vcf_file): # convert vcf file to bed format containing length of 60 bps
    vcf_data = pd.read_csv(vcf_file, sep="\t",header=None)
    vcf_data[0] = "chr" + vcf_data[0] # adding chr prefix for fetching seqs
    vcf_data[1] = vcf_data[1] - 1
    sample_name = vcf_data[2][0]
    bed_format = pd.concat([vcf_data[0], (vcf_data[1] - 5), (vcf_data[1] + 6)], axis=1)  # getting -30+ sequences
    bed_format.columns = ["chr", "start", "end"]
    bed_format.to_csv(f"sample_vcf_beds/{sample_name}_bed.txt", header=False, index=False, sep="\t")
    print(f"{sample_name} is converted to bed format!")
    return bed_format

# ADDING Mutational signature probabilities
def mutation_context(context,alt_point):
    ref_seq = context[1]
    alt_seq = alt_point
    five_prime = context[0]
    three_prime = context[2]
    return five_prime + "[" +  ref_seq + ">" + alt_seq + "]" + three_prime
def add_context(row):
    ref_point = row["ref"]
    alt_point = row["alt"]
    if  (ref_point == "G") or (ref_point == "A"):
        contxt = biocoder.revcompstr(row["sequence"][4:7])
        alt_point = biocoder.revcompstr(alt_point)
    else:
        contxt = row["sequence"][4:7]
    mutated_context = mutation_context(contxt,alt_point)
    return mutated_context
    # ------------- Featurizing SBS probs ----------
def sbs_columns(vcf_data,probs):
    vcf_data["MutationTypes"] = vcf_data.apply(add_context, axis=1)
    vcf_data = pd.merge(vcf_data, probs, how="left",on=["Sample Names","MutationTypes"])
    vcf_data.to_parquet(f"outputs/BreastCancer560_seq_probs.pqt",index=False)
    return vcf_data
def single_bed_to_fasta(bed_file,genome,id):
    # ref_genome = pyfastx.Fasta(f"genome_assembly/{genome}.fa.gz")
    ref_genome = Fasta(f"genome_assembly/{genome}.fa", sequence_always_upper=True)  # 1-based coordinates
    with open(bed_file, "r") as bed_data, open(f"sample_vcf_fasta/{id}_fasta.txt", "w") as output_file:
        for line in bed_data:
            chrom, start, end = line.strip().split("\t")[:3]
            start = int(start)
            end = int(end)
            sequence = ref_genome[chrom][(start):end].seq
            output_file.write(f">{chrom}:{start}-{end}\n")
            output_file.write(str(sequence) + "\n")
        print(f"{id} is converted to fasta format!")
def single_vcf_seq(vcf_file,fasta_file): # adding wild-type and altered 60 bps sequences into VCF
    vcf_data = pd.read_csv(vcf_file, sep="\t",header=None) # read original vcf file
    sample_name = vcf_data[2][0]
    with open(fasta_file) as f:  # fasta file to extract sequences
        fasta = f.readlines()
    locs = [y.split()[0].replace(">","").replace("(+)","") for x,y in enumerate(fasta) if x%2 == 0]
    seqs = [y.rstrip().upper().replace("N","") for x,y in enumerate(fasta) if x%2 != 0]
    vcf_60seq = pd.DataFrame({"region": locs, "sequence": seqs})
    vcf_data["sequence"] = vcf_60seq["sequence"]
    vcf_data["altered_seq"] = vcf_data.apply(altered_seq, axis=1)
    col = ["chr", "start", "Sample Names", "ref", "alt", "sequence", "altered_seq"]
    vcf_data.columns = col
    vcf_ID = vcf_data["Sample Names"][0]
    # vcf_data = sbs_columns(vcf_data, probs)
    vcf_data.to_csv(f"sample_vcf_seq_probs/{vcf_ID}_seq_probs.txt", sep="\t",index=False)
    print(f"{sample_name} is featured with to ref seq and alt seq!")
    return vcf_data

# Run vcf files
# probs = pd.read_csv("breast_cancer_samples/COSMIC_SBS96_Decomposed_Mutation_Probabilities.txt",sep="\t")

# For all vcf files in the global path names
def glob_vcf_to_bed(vcf_files): # convert vcf file to bed format containing length of 60 bps
    vcfs = glob.glob(vcf_files)
    for vcf in vcfs:
        single_vcf_to_bed(vcf)
    return "Operation successful"

def glob_bed_to_fasta(bed_files,genome):
    # ref_genome = pyfastx.Fasta(f"genome_assembly/{genome}.fa.gz")
    ref_genome = Fasta(f"genome_assembly/{genome}.fa", sequence_always_upper=True)  # 1-based coordinates
    beds = glob.glob(bed_files)
    for bed in beds:
        # vcf_name = bed.split("\\")[1].split("_")[0]
        vcf_name = "all_vcf"
        with open(bed, "r") as bed_file, open(f"sample_vcf_fasta/{vcf_name}_fasta.txt", "w") as output_file:
            for line in bed_file:
                chrom, start, end = line.strip().split("\t")[:3]
                start = int(start)
                end = int(end)
                sequence = ref_genome[chrom][start:end].seq
                output_file.write(f">{chrom}:{start}-{end}\n")
                output_file.write(str(sequence) + "\n")
            print(f"{vcf_name} is converted to fasta format!")
    return "Operation is done"

def glob_vcf_seq(vcf_files,fasta_files):
    vcfs = glob.glob(vcf_files)
    fastas = glob.glob(fasta_files)
    for vcf_file,fasta_file, in zip(vcfs,fastas):
        single_vcf_seq(vcf_file,fasta_file)
    return "Operation is done"

os.makedirs("sample_vcf_beds")

os.makedirs("sample_vcf_fasta")

os.makedirs("sample_vcf_seq_probs")
#1: bed format preparation
glob_vcf_to_bed("breast_cancer_samples/sample_vcfs/*.vcf")
glob_vcf_to_bed("BreastCancer560_all.txt")

#2: fasta: fetching sequence
glob_bed_to_fasta("sample_vcf_beds/*.txt","hg19")

glob_bed_to_fasta("sample_vcf_beds/PD10010a_bed.txt","hg19")

#3: vcf sequence and probs
glob_vcf_seq("breast_cancer_samples/sample_vcfs/*.vcf","sample_vcf_fasta/*.txt")
glob_vcf_seq("BreastCancer560_all.txt","sample_vcf_fasta/all_vcf_fasta.txt")
#------------------------------------------


BreastCancer560_prob = pd.read_csv("data/breastcancer560/breastcancer560_prob.txt",sep="\t")
vcf_file = pd.read_parquet("outputs/BreastCancer560_loss_gain_results_0.01.pqt")
vcf_results_sbs = sbs_columns(vcf_file,BreastCancer560_prob)

read_sbs = pd.read_parquet("outputs/BreastCancer560_results_0.01_probs.pqt")
read_sbs.iloc[:3,5:8]
read_sbs.iloc[:100].to_csv("outputs/sample_100FIRST_probs.csv",index=False)
read_sbs.sample(100).to_csv("outputs/sample_100_probs.csv",index=False)

################### FOR MEMORY AND DISK-SPACE REDUCTION

all_files = pd.read_csv("sample_vcf_seq_probs/all_seq_probs.csv")

# NEW COLUMNS
all_files["iref"] = [bio.seqtoi(x) for x in all_files['ref']]
all_files["ialt"] = [bio.seqtoi(x) for x in all_files['alt']]
all_files["isequence"] = [bio.seqtoi(x) for x in all_files['sequence']]
all_files["ialtered_seq"] = [bio.seqtoi(x) for x in all_files['altered_seq']]

all_files["Sample Names"] = all_files["Sample Names"].astype("category")
all_files["chr"] = all_files["chr"].astype("category")

# DOWNCASTING
all_files["start"] = pd.to_numeric(all_files["start"],downcast= "unsigned")
all_files["iref"] = pd.to_numeric(all_files["iref"],downcast= "unsigned")
all_files["ialt"] = pd.to_numeric(all_files["ialt"],downcast= "unsigned")
all_files["isequence"] = pd.to_numeric(all_files["isequence"],downcast= "unsigned")
all_files["ialtered_seq"] = pd.to_numeric(all_files["ialtered_seq"],downcast= "unsigned")

all_files_shrunk = all_files[["chr","start","Sample","isequence","ialtered_seq"]]
all_files_shrunk.info(memory_usage="deeper")
dtype_dict = all_files_shrunk.dtypes.to_dict()

all_files_shrunk.to_parquet("all_files_int.pqt",index=False)
all_files_shrunk.to_parquet("testall.gzip",compression="gzip")

s = time.time()
c = pd.read_csv("sample_vcf_seq_probs/all_files_integer_pred.csv",dtype=dtype_dict)
f = time.time()

all_files_shrunk.to_pickle("sample_vcf_seq_probs/test.pkl")
df_pqt=pd.read_parquet('outputs/pred_results_ch2_new/chunk#0/pred_chunk#0_AR.pqt')
df_pqt2=pd.read_parquet("outputs/pred_results_ch3_old/chunk#0/pred_chunk#0_AR.pqt")
df_pqt.describe()
df_pqt.to_parquet("all_files_int.pqt",index=False)
df_pqt.info(memory_usage="deeper")
df_pqt2[df_pqt2["p_value"] != df_pqt2["p_value"]]
df_pqt[df_pqt2["p_value"] != df_pqt2["p_value"]]

groups  = df_pqt.groupby('Sample')
i = 1
sample_dict = {}
max(sample_dict, key=sample_dict.get)
samples_info = pd.DataFrame([sample_dict]).T
samples_info.sort_values(by=0,ascending=False).to_csv("sample_size.csv")

for name, group in groups:
    # print(f"Sample: {name}\n{group}\n")
    print(i)
    sample_dict[name] = len(group)
    i += 1

chunk_size = 20000
for j,i in enumerate(range(0, 3479652, chunk_size)):
    vcf_data_chunk = pd.read_parquet("sample_vcf_seq_probs/all_files_int.pqt")[i:i + chunk_size]
    vcf_ID = "chunk#" + str(j)
    print(vcf_ID)


