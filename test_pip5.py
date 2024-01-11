import concurrent.futures
import glob
import pandas as pd
import os
import pickle
import bio
import scipy.stats
import numpy as np
import multiprocessing
import time
def mypredict(seq1, seq2, params,cov_matrix,TF_name):
    ref = bio.nonr_olig_freq([bio.seqtoi(seq1)],6,nonrev_list)  # from N
    mut = bio.nonr_olig_freq([bio.seqtoi(seq2)],6,nonrev_list)  # to N
    diff_count = mut - ref # c': count matrix
    diff = np.dot(diff_count, params) # c'Bhat --> the difference in binding affinity
    # print(f"Process {os.getpid()} and {TF_name} diff score aka (wildBeta-mutatedBeta) (c'): ",diff)
    SE = np.sqrt((np.dot(diff_count,cov_matrix) * diff_count).sum(axis=1)) # 2080*2080 X 2080*1 = 2080*1 # a scalar value Standart error
    t =  diff/ SE # t-statistic : c'Bhat / sqrt(c'*covBhat*c)
    p_val = scipy.stats.norm.sf(abs(t))*2 # follows t-distribution
    statdict = {"diff":diff[0], "t":t[0], "p_value":p_val[0] }
    return statdict
def pred_vcf(param,TF_name,vcf_seq,vcf_ID):
    stat_df = vcf_seq.apply(lambda x: mypredict(x["sequence"],x["altered_seq"],param[0],param[1],TF_name),
                                                   axis=1,result_type="expand")
    pred_df = pd.concat([vcf_seq,stat_df],axis=1)
    pred_df.to_csv(f"outputs/pred_results/{vcf_ID}/pred_{vcf_ID}_{TF_name}.csv", index=False)
def process_vcf(param, TF_name, vcf_seq,vcf_ID):
    start = time.time()
    current_process = multiprocessing.current_process().name
    print(f"\nProcess {current_process} is analyzing VCF file: {vcf_ID}, TF_model: {TF_name}")
    pred_vcf(param, TF_name, vcf_seq,vcf_ID)
    print(f"\nProcess {current_process} finished analyzing VCF file: {vcf_ID}, TF_model: {TF_name}")
    end = time.time()
    print(f"\nTime -----------:{end-start}")
#219 secs full time,  215secs single without jobs
#226 secs , 220
#218 secs / 225 second run for concurrentproc
nonrev_list = bio.gen_nonreversed_kmer(6)  # 2080 features (6-mer DNA)

def main(num_proc):

    vcf_seqs = glob.glob("breast_cancer_samples/sample_vcf_seq_probs/*csv")  # I have 21 vcf files to be analyzed (large file)

    # Listing pre-computed-pred files
    params = glob.glob("outputs/params/*.pkl")
    param_dict = {}  # store pre-computed parameters
    for param in params:  # I have 50 pre-computed parameters of models
        with open(param, "rb") as file:
            tuple_param = pickle.load(file)
            coef = np.array(tuple_param[1], dtype=np.float32)
            covar = np.array(tuple_param[3], dtype=np.float32)
            param_dict[param] = [coef, covar]

    if not os.path.exists("outputs/pred_results"):
        os.makedirs("outputs/pred_results")

    start_time = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_proc) as executor:
        futures = []
        for vcf_seq in vcf_seqs:
            vcf_ID = os.path.splitext(os.path.basename(vcf_seq))[0].split("_")[0]
            vcf_data = pd.read_csv(vcf_seq)

            if not os.path.exists(f"outputs/pred_results/{vcf_ID}"):
                os.makedirs(f"outputs/pred_results/{vcf_ID}")
                print(f"outputs/pred_results/{vcf_ID} folder is created!")

            for param_file, param_data in param_dict.items():
                TF_name = param_file.split("_")[-1].split(".")[0]
                futures.append(executor.submit(process_vcf, param_data, TF_name, vcf_data, vcf_ID))

        # for future in concurrent.futures.as_completed(futures):
        #     try:
        #         result = future.result()
        #     except Exception as e:
        #         print(f"Error: {e}")

    finish_time = time.perf_counter()
    print("Finished!")
    print("Elapsed time during the whole program in seconds:", finish_time - start_time)


if __name__ == "__main__":
    print("Predictions Started!")
    main(12)

###################################################################################################
# @profile
# def main(n_proc):
#     # vcf_seqs = glob.glob("breast_cancer_samples/sample_vcf_seq_probs/*.csv")[:2]  # I have 21 vcf files to be analyzed
#     vcf_seqs = glob.glob("breast_cancer_samples/sample_vcf_seq_probs/*csv")[:1]  # I have 21 vcf files to be analyzed
#     # Listing pre-computed-pred files
#     params = glob.glob("outputs/params/*.pkl")[:12]
#     param_dict = {}  # store pre-computed parameters
#     for param in params:  # I have 50 pre-computed parameters of models
#         with open(param, "rb") as file:
#             tuple_param = pickle.load(file)
#             coef = np.array(tuple_param[1], dtype=np.float32)
#             covar = np.array(tuple_param[3], dtype=np.float32)
#             param_dict[param] = [coef, covar]
#     # dict(list(param_dict.items())[chunk_size:chunksize])
#     start_time = time.perf_counter()
#     with multiprocessing.Pool(processes=n_proc) as pool:
#         for vcf_seq in vcf_seqs:
#             vcf_ID = os.path.splitext(os.path.basename(vcf_seq))[0].split("_")[0]
#             vcf_data = pd.read_csv(vcf_seq)
#             if not os.path.exists(f"outputs/pred_results/{vcf_ID}"):
#                 os.makedirs(f"outputs/pred_results/{vcf_ID}")
#                 print(f"outputs/pred_results/{vcf_ID} folder is created!")
#
#             for param_file, param_data in param_dict.items():
#                 TF_name = param_file.split("_")[-1].split(".")[0]
#                 pool.apply_async(process_vcf, args=(param_data, TF_name, vcf_data, vcf_ID))
#         pool.close()
#         pool.join()
#     finish_time = time.perf_counter()
#     print("\nFinished!")
#     print("\nElapsed time during the whole program in seconds:", finish_time - start_time)


###################################################################################################
# vcf_seqs = glob.glob("breast_cancer_samples/sample_vcf_seq_probs/*csv")[:1]  # I have 21 vcf files to be analyzed
# # Listing pre-computed-pred files
# params = glob.glob("outputs/params/*.pkl")[:12]
# param_dict = {}  # store pre-computed parameters
# for param in params:  # I have 50 pre-computed parameters of models
#     with open(param, "rb") as file:
#         tuple_param = pickle.load(file)
#         coef = np.array(tuple_param[1], dtype=np.float32)
#         covar = np.array(tuple_param[3], dtype=np.float32)
#         param_dict[param] = [coef,covar]
#
#
# # dict(list(param_dict.items())[chunk_size:chunksize])
# if __name__ == "__main__":
#     # vcf_seqs = glob.glob("breast_cancer_samples/sample_vcf_seq_probs/*.csv")[:2]  # I have 21 vcf files to be analyzed
#     start_time = time.perf_counter()
#     with multiprocessing.Pool(processes=12) as pool:
#         for vcf_seq in vcf_seqs:
#             vcf_ID = os.path.splitext(os.path.basename(vcf_seq))[0].split("_")[0]
#             vcf_data = pd.read_csv(vcf_seq)
#             if not os.path.exists(f"outputs/pred_results/{vcf_ID}"):
#                 os.makedirs(f"outputs/pred_results/{vcf_ID}")
#                 print(f"outputs/pred_results/{vcf_ID} folder is created!")
#
#             for param_file, param_data in param_dict.items():
#                 TF_name = param_file.split("_")[-1].split(".")[0]
#                 pool.apply_async(process_vcf, args=(param_data, TF_name, vcf_data, vcf_ID))
#         pool.close()
#         pool.join()
#     finish_time = time.perf_counter()
#     print("Finished!")
#     print("Elapsed time during the whole program in seconds:", finish_time-start_time)
# # def process_vcf_wrapper(args):
# #     param_data, TF_name, vcf_seq, vcf_ID = args
# #     process_vcf(param_data, TF_name, vcf_seq, vcf_ID)
#
# # if __name__ == "__main__":
# #     with multiprocessing.Pool() as pool:
# #         jobs = []
# #         for vcf_seq_file in vcf_seqs:
# #             vcf_ID = vcf_seq_file.split("\\")[1].split("_")[0]
# #             vcf_seq = pd.read_csv(vcf_seq_file)
# #
# #             if not os.path.exists(f"outputs/pred_results/{vcf_ID}"):
# #                 os.makedirs(f"outputs/pred_results/{vcf_ID}")
# #                 print(f"outputs/pred_results/{vcf_ID} folder is created!")
# #
# #             for param_file, param_data in param_dict.items():
# #                 TF_name = param_file.split("_")[-1].split(".")[0]
# #                 jobs.append((param_data, TF_name, vcf_seq, vcf_ID))
# #         pool.map(process_vcf_wrapper, jobs)
# #         pool.close()
# #         pool.join()