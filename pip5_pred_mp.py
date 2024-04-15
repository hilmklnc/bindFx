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
import pyarrow.parquet as pq

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
    chunk_params = 10
    n_chunks = (3479651 // chunk_size)
    total_iterations =  n_chunks * 403

    for i in range(0,403,chunk_params):
        params = glob.glob("data/params403/*.pkl")[i: i + chunk_params]
        param_dict = {}  # store pre-computed parameters
        for param in params:  # I have 50 pre-computed parameters of models
            with open(param, "rb") as file:
                tuple_param = pickle.load(file)
                param_dict[param] = [tuple_param[0], tuple_param[1]]
        def my_callback(result):
            print(f"\n[Worker {result[3].split("-")[1]}]-Elapsed time for {result[2]} in {result[1]}:------------{round(result[0],3)} sec")

        with multiprocessing.Pool(processes=n_proc) as pool:

            with pq.ParquetFile(vcf_filename) as parquet_file:

                for j,chunk in enumerate(parquet_file.iter_batches(batch_size=chunk_size)):
                    vcf_data_chunk = chunk.to_pandas()
                    vcf_ID = "chunk#" + str(j)
                    if (not os.path.exists(f"outputs/pred_results/{vcf_ID}")) and (i == 0):
                        os.makedirs(f"outputs/pred_results/{vcf_ID}")
                        print(f"outputs/pred_results/{vcf_ID} folder is created!")

                    for param_file, param_data in param_dict.items():
                        TF_name = param_file.split("_")[-1].split(".")[0]
                        pool.apply_async(process_vcf, args=(param_data, TF_name,  vcf_data_chunk, vcf_ID)
                                         ,callback=my_callback)
            pool.close()
            pool.join()

    finish_time = time.time()

    print("\nFinished!")
    print(f"\nElapsed time during the whole program in seconds: {(finish_time - start_time)/3600} h\n")

if __name__ == "__main__":

    print("########### Predictions Started! ###########\n")
    main(7)
    print("---------------------Ended---------------------")
