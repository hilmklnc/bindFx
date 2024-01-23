import glob
import os
import pickle
import biocoder
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
    sequences = vcf_data['isequence'].to_list()
    altered_sequences = vcf_data['ialtered_seq'].to_list()

    ref = biocoder.nonr_olig_freq(sequences)
    mut = biocoder.nonr_olig_freq(altered_sequences)
    diff_count = (mut - ref).to_numpy()
    diff = np.dot(diff_count, TF_param[0])
    SE = np.sqrt(np.abs((np.dot(diff_count,  TF_param[1]) * diff_count).sum(axis=1)))
    t = diff / SE
    p_val = scipy.stats.norm.sf(np.abs(t)) * 2
    pred_vcf = vcf_data.assign(diff=diff, t=t, p_value=p_val)
    pred_vcf.to_parquet(f"outputs/pred_results/{vcf_ID}/pred_{vcf_ID}_{TF_name}.pqt", index=False)

def process_vcf(param, TF_name, vcf_data,vcf_ID):
    current_process = multiprocessing.current_process().name
    start = time.time()
    pred_vcf(param, TF_name, vcf_data,vcf_ID)
    end = time.time()
    return end-start,vcf_ID,TF_name,current_process

def main(n_proc):

    start_time = time.time()
    vcf_filename = "data/breastcancer560/all_files_int.pqt"
    chunk_size = 20000
    params = glob.glob("data/params403/*.pkl")

    param_dict = {}  # store pre-computed parameters
    for param in params:  # I have 50 pre-computed parameters of models
        with open(param, "rb") as file:
            tuple_param = pickle.load(file)
            param_dict[param] = [tuple_param[0], tuple_param[1]]
    n_chunks =  (3479651 // chunk_size)
    total_iterations =  n_chunks * len(param_dict)

    with tqdm(total=total_iterations) as pbar:
        def my_callback(result):
            pbar.update()
            pbar.set_description(f"\nW{result[3].split("-")[1]}-Elapsed time for {result[2]} in {result[1]}: {round(result[0],3)}\n")

        with multiprocessing.Pool(processes=n_proc) as pool:

            for j, i in enumerate(range(0, 3479651, chunk_size)):
                vcf_data_chunk = pd.read_parquet(vcf_filename)[i:i + chunk_size]
                vcf_ID = "chunk#" + str(j)
                if not os.path.exists(f"outputs/pred_results/{vcf_ID}"):
                    os.makedirs(f"outputs/pred_results/{vcf_ID}")
                    print(f"outputs/pred_results/{vcf_ID} folder is created!")

                for param_file, param_data in param_dict.items():
                    TF_name = param_file.split("_")[-1].split(".")[0]
                    pool.apply_async(process_vcf, args=(param_data, TF_name,  vcf_data_chunk, vcf_ID)
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
    main(7)
    logging.info("---------------------Ended---------------------")


