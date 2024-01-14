import glob
import os
import pickle
import bio
import numpy as np
import pandas as pd
import scipy.stats
import multiprocessing
import time
import warnings
import logging
from tqdm import tqdm

warnings.filterwarnings("ignore")

def pred_vcf(TF_param, TF_name, vcf_data, vcf_ID):
    sequences = [bio.seqtoi(x) for x in vcf_data['sequence']]
    altered_sequences =[bio.seqtoi(y) for y in vcf_data['altered_seq']]
    ref = bio.nonr_olig_freq(sequences)
    mut = bio.nonr_olig_freq(altered_sequences)
    diff_count = (mut - ref).to_numpy()
    diff = np.dot(diff_count, TF_param[0])
    SE = np.sqrt(np.abs((np.dot(diff_count,  TF_param[1]) * diff_count).sum(axis=1)))
    t = diff / SE
    p_val = scipy.stats.norm.sf(np.abs(t)) * 2
    pred_vcf = vcf_data.assign(diff=diff, t=t, p_value=p_val)
    # Save the DataFrame to a CSV file
    pred_vcf.to_csv(f"outputs/pred_results/{vcf_ID}/pred_{vcf_ID}_{TF_name}.csv", index=False)

def process_vcf(param, TF_name, vcf_data,vcf_ID):
    start = time.time()
    current_process = multiprocessing.current_process().name
    pred_vcf(param, TF_name, vcf_data,vcf_ID)
    end = time.time()
    return end-start,vcf_ID,TF_name,current_process

def main(n_proc):
    vcf_seq_files = glob.glob("breast_cancer_samples/sample_vcf_seq_probs/*csv") # I have 21 vcf files to be analyzed
    # Listing pre-computed-pred file
    params = glob.glob("outputs/params/*.pkl")
    param_dict = {}  # store pre-computed parameters
    for param in params:  # I have 50 pre-computed parameters of models
        with open(param, "rb") as file:
            tuple_param = pickle.load(file)
            coef = np.array(tuple_param[1], dtype=np.float32)
            covar = np.array(tuple_param[3], dtype=np.float32)
            param_dict[param] = [coef, covar]

    start_time = time.time()
    total_iterations = len(vcf_seq_files) * len(param_dict)
    with tqdm(total=total_iterations, unit=" TF-model",colour = "#004000") as pbar:
        def my_callback(result):
            pbar.update()
            pbar.set_description(f"W{result[3].split("-")[1]}-Elapsed time for {result[2]} in {result[1]}: {round(result[0],3)}")

        with multiprocessing.Pool(processes=n_proc) as pool:

            for vcf_file in vcf_seq_files:
                vcf_ID = os.path.splitext(os.path.basename(vcf_file))[0].split("_")[0]
                if not os.path.exists(f"outputs/pred_results/{vcf_ID}"):
                    os.makedirs(f"outputs/pred_results/{vcf_ID}")
                    print(f"outputs/pred_results/{vcf_ID} folder is created!")
                vcf_data = pd.read_csv(vcf_file)

                for param_file, param_data in param_dict.items():
                    TF_name = param_file.split("_")[-1].split(".")[0]
                    pool.apply_async(process_vcf, args=(param_data, TF_name,  vcf_data, vcf_ID)
                                     ,callback=my_callback)

            pool.close()
            pool.join()

    finish_time = time.time()

    logging.info("Finished!\n")
    logging.info(f"Elapsed time during the whole program in seconds: {finish_time - start_time}\n")

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                        format='%(asctime)s [%(levelname)s] - %(message)s')

    logging.info("########### Predictions Started! ###########\n")
    main(os.cpu_count())
    logging.info("---------------------Ended---------------------")


