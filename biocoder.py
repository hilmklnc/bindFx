# -*- coding: utf-8 -*-
"""
Bio file to be used in bindFx training prediction file
"""
import pandas as pd
import numpy as np

# all permutations are already reverse-deleted
# all sequences are represented in binary

nucleotides = {'A':0,'C':1,'G':2,'T':3}
numtonuc = {0:'A',1:'C',2:'G',3:'T'}
complement = {0:3,3:0,1:2,2:1}

def window(fseq, window_size):
    for i in range(len(fseq) - window_size + 1):
        yield fseq[i:i+window_size]

# return the first or the last number representation
def seqpos(kmer,last):
    return 1 <<  (1 + 2 * kmer) if last else 1 << 2 * kmer;

def seq_permutation(seqlen):
    return (range(seqpos(seqlen,False),seqpos(seqlen,True)))

def gen_nonreversed_kmer(k):
    nonrevk = list()
    for i in range(seqpos(k,False),seqpos(k,True)):
        if i <= revcomp(i):
            nonrevk.append(i)
    return nonrevk

def itoseq(seqint):
    if type(seqint) is not int:
        return seqint
    seq = ""
    mask = 3
    copy = int(seqint) # prevent changing the original value
    while(copy) != 1:
        seq = numtonuc[copy&mask] + seq
        copy >>= 2
        if copy == 0:
            print("Could not find the append-left on the input sequence")
            return 0
    return seq

def seqtoi(seq,gappos=0,gapsize=0):
    # due to various seqlengths, this project always needs append 1 to the left
    binrep = 1
    gaps = range(gappos,gappos+gapsize)
    for i in range(0,len(seq)):
        if i in gaps:
            continue
        binrep <<= 2
        binrep |= nucleotides[seq[i]]
    return binrep

def revcomp(seqbin):
    rev = 1
    mask = 3
    copy = int(seqbin)

    while copy != 1:
        rev <<= 2
        rev |= complement[copy&mask]
        copy >>= 2
        if copy == 0:
            print("Could not find the append-left on the input sequence")
            return 0
    return rev

def revcompstr(seq):
    rev = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return "".join([rev[base] for base in reversed(seq)])

def insert_pos(seqint,base,pos): # pos is position from the right
    return ((seqint << 2) & ~(2**(2*pos+2)-1)) | ((seqint & 2**(2*pos)-1) | (nucleotides[base] << pos*2))
    #return (seqint << 2) | (seqint & 2**pos-1) & ~(3 << (pos*2)) | (nucleotides[base] << pos*2)

# this function already counts without its reverse complement,
# i.e. oligfreq + reverse merge in the original R code
# Input: panda list and kmer length
# Output: oligonucleotide count with reverse removed

# arrange the dtypes in the structures according to specific data
# def nonr_olig_freq2(seqtbl, kmer, nonrev_list):
#     rightseparator = kmer
#     leftseparator = rightseparator
#
#     # Use numpy for faster array operations
#     seqint_arr = np.array(seqtbl)
#     # olig_df = np.zeros((len(nonrev_list), len(seqtbl)))
#     olig_df = np.zeros((len(seqtbl), len(nonrev_list)))
#
#     mask = (4 ** kmer) - 1
#     for i in range(len(seqtbl)):
#         cpy = seqint_arr[i]
#         while cpy > mask:
#             cur = cpy & mask
#             right = cur & ((4 ** rightseparator) - 1)
#             left = (cur >> (2 * leftseparator)) << (2 * rightseparator)
#             seqint = left | right
#
#             r = (1 << (2 * kmer)) | seqint  # append 1
#             rc = revcomp(r)
#
#             r = r if r < rc else rc  # Use conditional operator
#             # Use numpy indexing for faster updates
#             # olig_df[nonrev_list.index(r), i] += 1
#             olig_df[i, nonrev_list.index(r)] += 1
#             cpy >>= 2
#
#     return pd.DataFrame(olig_df, columns=nonrev_list)
def nonr_olig_freq(seqtbl, kmer=6):
    nonrev_list = gen_nonreversed_kmer(kmer)
    rightseparator = kmer
    leftseparator = rightseparator
    # Use numpy for faster array operations
    seqint_arr = np.array(seqtbl) # adjust dtypes if it is necessary
    olig_df = np.zeros((len(seqtbl), len(nonrev_list)))
    mask = (4 ** kmer) - 1
    for i, cpy in enumerate(seqint_arr):
        while cpy > mask:
            cur = cpy & mask
            right = cur & ((4 ** rightseparator) - 1)
            left = (cur >> (2 * leftseparator)) << (2 * rightseparator)
            seqint = left | right

            r = (1 << (2 * kmer)) | seqint  # append 1
            rc = revcomp(r)
            r = r if r < rc else rc  # Use conditional operator
            # Use numpy indexing for faster updates
            olig_df[i, nonrev_list.index(r)] += 1
            cpy >>= 2
    return pd.DataFrame(olig_df, columns=nonrev_list)

