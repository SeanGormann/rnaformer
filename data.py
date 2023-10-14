import pandas as pd
import os, gc
import numpy as np
from sklearn.model_selection import KFold
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

nfolds = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
bs = 128


def count_unique_kmers(sequences, k):
    kmers = set()
    for seq in sequences:
        kmers.update([seq[i:i+k] for i in range(len(seq) - k + 1)])
    return len(kmers), kmers

#sequences = df['sequence'].values  # Assuming df is your DataFrame and it has a column named 'sequence'
#k = 3  # The length of the k-mers
#num_embeddings, kmers = count_unique_kmers(sequences, k)




class RNA_Dataset(Dataset):
    def __init__(self, df, mode='train', seed=2023, fold=0, nfolds=4,
                 mask_only=False, **kwargs):
        self.seq_map = {'A':0,'C':1,'G':2,'U':3}
        self.Lmax = 206
        df['L'] = df.sequence.apply(len)
        df_2A3 = df.loc[df.experiment_type=='2A3_MaP']
        df_DMS = df.loc[df.experiment_type=='DMS_MaP']

        split = list(KFold(n_splits=nfolds, random_state=seed,
                shuffle=True).split(df_2A3))[fold][0 if mode=='train' else 1]
        df_2A3 = df_2A3.iloc[split].reset_index(drop=True)
        df_DMS = df_DMS.iloc[split].reset_index(drop=True)

        #m = (df_2A3['SN_filter'].values > 0) & (df_DMS['SN_filter'].values > 0)
        m = (df_2A3['signal_to_noise'].values >= 0.5) & (df_DMS['signal_to_noise'].values >= 0.5)
        df_2A3 = df_2A3.loc[m].reset_index(drop=True)
        df_DMS = df_DMS.loc[m].reset_index(drop=True)

        self.seq = df_2A3['sequence'].values
        self.L = df_2A3['L'].values

        self.react_2A3 = df_2A3[[c for c in df_2A3.columns if \
                                 'reactivity_0' in c]].values
        self.react_DMS = df_DMS[[c for c in df_DMS.columns if \
                                 'reactivity_0' in c]].values
        self.react_err_2A3 = df_2A3[[c for c in df_2A3.columns if \
                                 'reactivity_error_0' in c]].values
        self.react_err_DMS = df_DMS[[c for c in df_DMS.columns if \
                                'reactivity_error_0' in c]].values
        self.sn_2A3 = df_2A3['signal_to_noise'].values
        self.sn_DMS = df_DMS['signal_to_noise'].values
        self.mask_only = mask_only
        self.mode = mode

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        seq = self.seq[idx]
        if self.mask_only:
            mask = torch.zeros(self.Lmax, dtype=torch.bool)
            mask[:len(seq)] = True
            return {'mask':mask},{'mask':mask}
        seq = [self.seq_map[s] for s in seq]
        seq = np.array(seq)
        mask = torch.zeros(self.Lmax, dtype=torch.bool)
        mask[:len(seq)] = True
        seq = np.pad(seq,(0,self.Lmax-len(seq)))

        react = torch.from_numpy(np.stack([self.react_2A3[idx],
                                           self.react_DMS[idx]],-1))
        react_err = torch.from_numpy(np.stack([self.react_err_2A3[idx],
                                               self.react_err_DMS[idx]],-1))
        sn = torch.FloatTensor([self.sn_2A3[idx],self.sn_DMS[idx]])

        
        """#### Masking sequences to UNK ####
        if self.mode == 'train':  
            mask_length = 4  # Number of contiguous base pairs to mask

            dynamic_mask_rate = np.random.uniform(low=0.01, high=0.05) 
            num_positions_to_mask = int((len(seq) - mask_length + 1) * dynamic_mask_rate)
            mask_positions = np.random.choice(len(seq) - mask_length + 1, num_positions_to_mask, replace=False)
            
            global_mask_rate = 0.4
            if np.random.rand() < global_mask_rate:
                # Apply the mask
                for start in mask_positions:
                    seq[start:start + mask_length] = self.seq_map['UNK']
                    react[start:start + mask_length] = float('nan')"""

        return {'seq':torch.from_numpy(seq), 'mask':mask}, \
               {'react':react, 'react_err':react_err,
                'sn':sn, 'mask':mask}



class RNA_SE_Dataset(Dataset):
    def __init__(self, df, mode='train', seed=2023, fold=0, nfolds=4,
                 mask_only=False, **kwargs):
        self.seq_map = {'A':0,'C':1,'G':2,'U':3, '<s>':4, '</s>':5}  # Added start and end tokens
        self.Lmax = 206
        df['L'] = df.sequence.apply(len)
        df_2A3 = df.loc[df.experiment_type=='2A3_MaP']
        df_DMS = df.loc[df.experiment_type=='DMS_MaP']

        split = list(KFold(n_splits=nfolds, random_state=seed,
                shuffle=True).split(df_2A3))[fold][0 if mode=='train' else 1]
        df_2A3 = df_2A3.iloc[split].reset_index(drop=True)
        df_DMS = df_DMS.iloc[split].reset_index(drop=True)

        #m = (df_2A3['SN_filter'].values > 0) & (df_DMS['SN_filter'].values > 0)
        m = (df_2A3['signal_to_noise'].values >= 1) & (df_DMS['signal_to_noise'].values >= 1)
        df_2A3 = df_2A3.loc[m].reset_index(drop=True)
        df_DMS = df_DMS.loc[m].reset_index(drop=True)

        self.seq = df_2A3['sequence'].values
        self.L = df_2A3['L'].values

        self.react_2A3 = df_2A3[[c for c in df_2A3.columns if \
                                 'reactivity_0' in c]].values
        self.react_DMS = df_DMS[[c for c in df_DMS.columns if \
                                 'reactivity_0' in c]].values
        self.react_err_2A3 = df_2A3[[c for c in df_2A3.columns if \
                                 'reactivity_error_0' in c]].values
        self.react_err_DMS = df_DMS[[c for c in df_DMS.columns if \
                                'reactivity_error_0' in c]].values
        self.sn_2A3 = df_2A3['signal_to_noise'].values
        self.sn_DMS = df_DMS['signal_to_noise'].values
        self.mask_only = mask_only

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        seq = self.seq[idx]
        if self.mask_only:
            mask = torch.zeros(self.Lmax + 2, dtype=torch.bool)
            mask[:len(seq) + 2] = True
            return {'mask': mask}, {'mask': mask}

        seq = [self.seq_map['<s>']] + [self.seq_map[s] for s in seq] + [self.seq_map['</s>']]
        seq = np.array(seq)
        mask = torch.zeros(self.Lmax + 2, dtype=torch.bool)
        mask[:len(seq)] = True
        seq = np.pad(seq, (0, self.Lmax + 2 - len(seq)))

        react_2A3 = np.pad(self.react_2A3[idx], (1, 1), mode='constant', constant_values=np.nan)
        react_DMS = np.pad(self.react_DMS[idx], (1, 1), mode='constant', constant_values=np.nan)

        react = torch.from_numpy(np.stack([react_2A3, react_DMS], -1))
        react_err = torch.from_numpy(np.stack([self.react_err_2A3[idx], self.react_err_DMS[idx]], -1))
        sn = torch.FloatTensor([self.sn_2A3[idx], self.sn_DMS[idx]])

        # Expanding sn to match the shape of react and applying the mask
        #sn = sn.unsqueeze(1).expand(-1, self.Lmax + 2, -1)  # Adjusted length to match react
        #sn = sn * mask.unsqueeze(-1).float()  # Applying the mask

        return {'seq': torch.from_numpy(seq), 'mask': mask}, \
              {'react': react, 'react_err': react_err, 'sn': sn, 'mask': mask}







class RNA_KMERS_Dataset(Dataset):
    def __init__(self, df, mode='train', seed=2023, fold=0, nfolds=4,
                 mask_only=False, k=3, kmers=None, **kwargs):
        self.k = k
        self.seq_map = {'A':0,'C':1,'G':2,'U':3}  # Added start and end tokens
        self.kmer_map = {kmer: idx for idx, kmer in enumerate(kmers)}
        self.kmer_map['UNK'] = len(self.kmer_map)
        self.Lmax = 206
        df['L'] = df.sequence.apply(len)
        df_2A3 = df.loc[df.experiment_type=='2A3_MaP']
        df_DMS = df.loc[df.experiment_type=='DMS_MaP']

        split = list(KFold(n_splits=nfolds, random_state=seed,
                shuffle=True).split(df_2A3))[fold][0 if mode=='train' else 1]
        df_2A3 = df_2A3.iloc[split].reset_index(drop=True)
        df_DMS = df_DMS.iloc[split].reset_index(drop=True)

        m = (df_2A3['SN_filter'].values > 0) & (df_DMS['SN_filter'].values > 0)
        #m = (df_2A3['signal_to_noise'].values >= 0.7) & (df_DMS['signal_to_noise'].values >= 0.7)
        df_2A3 = df_2A3.loc[m].reset_index(drop=True)
        df_DMS = df_DMS.loc[m].reset_index(drop=True)

        self.seq = df_2A3['sequence'].values
        self.L = df_2A3['L'].values

        self.react_2A3 = df_2A3[[c for c in df_2A3.columns if \
                                 'reactivity_0' in c]].values
        self.react_DMS = df_DMS[[c for c in df_DMS.columns if \
                                 'reactivity_0' in c]].values
        self.react_err_2A3 = df_2A3[[c for c in df_2A3.columns if \
                                 'reactivity_error_0' in c]].values
        self.react_err_DMS = df_DMS[[c for c in df_DMS.columns if \
                                'reactivity_error_0' in c]].values
        self.sn_2A3 = df_2A3['signal_to_noise'].values
        self.sn_DMS = df_DMS['signal_to_noise'].values
        self.mask_only = mask_only


    def encode_kmers(self, kmers):
        return [self.kmer_map.get(kmer, self.kmer_map['UNK']) for kmer in kmers]

    """def generate_kmers(self, seq, k):
        kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
        for i in range(1, k):  # Handling the last k-1 characters of the sequence
            kmers.append(seq[-i:] + 'N' * (k - i))  # Padding with 'N' to make it k characters long
        return kmers"""

    def generate_kmers(self, seq, k):
        kmers = []
        half_k = (k - 1) // 2  # Calculate the number of neighbors on each side
        padded_seq = 'N' * half_k + seq + 'N' * half_k

        for i in range(half_k, len(seq) + half_k):
            kmer = padded_seq[i - half_k: i + half_k + 1]
            kmers.append(kmer)

        return kmers

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        # Getting the sequence
        original_seq = self.seq[idx]
        seq = [self.seq_map[s] for s in original_seq]
        seq = np.array(seq)
        seq_mask = torch.zeros(self.Lmax, dtype=torch.bool)
        seq_mask[:len(seq)] = True
        seq = np.pad(seq, (0, self.Lmax - len(seq)))

        # Getting the kmers
        kmers = self.generate_kmers(original_seq, self.k)
        encoded_kmers = self.encode_kmers(kmers)
        encoded_kmers = torch.tensor(encoded_kmers)
        padded_kmers = np.pad(encoded_kmers, (0, self.Lmax - len(encoded_kmers)))
        kmer_mask = torch.zeros(self.Lmax, dtype=torch.bool)
        kmer_mask[:len(encoded_kmers)] = True

        if self.mask_only:
            return {'mask': seq_mask, 'kmer_mask': kmer_mask}, {'mask': seq_mask, 'kmer_mask': kmer_mask}

        react = torch.from_numpy(np.stack([self.react_2A3[idx][:len(padded_kmers)],
                                           self.react_DMS[idx][:len(padded_kmers)]], -1))
        react_err = torch.from_numpy(np.stack([self.react_err_2A3[idx][:len(padded_kmers)],
                                               self.react_err_DMS[idx][:len(padded_kmers)]], -1))
        sn = torch.FloatTensor([self.sn_2A3[idx], self.sn_DMS[idx]])

        return {'seq': torch.from_numpy(seq), 'mask': seq_mask,
                'kmers': torch.from_numpy(padded_kmers), 'kmer_mask': kmer_mask}, \
               {'react': react, 'react_err': react_err, 'sn': sn, 'mask': seq_mask}




class LenMatchBatchSampler(torch.utils.data.BatchSampler):
    def __iter__(self):
        buckets = [[]] * 100
        yielded = 0

        for idx in self.sampler:
            s = self.sampler.data_source[idx]
            if isinstance(s,tuple): L = s[0]["mask"].sum()
            else: L = s["mask"].sum()
            L = max(1,L // 16)
            if len(buckets[L]) == 0:  buckets[L] = []
            buckets[L].append(idx)

            if len(buckets[L]) == self.batch_size:
                batch = list(buckets[L])
                yield batch
                yielded += 1
                buckets[L] = []

        batch = []
        leftover = [idx for bucket in buckets for idx in bucket]

        for idx in leftover:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yielded += 1
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yielded += 1
            yield batch

def dict_to(x, device='cuda'):
    return {k:x[k].to(device) for k in x}

def to_device(x, device='cuda'):
    return tuple(dict_to(e,device) for e in x)

class DeviceDataLoader:
    def __init__(self, dataloader, device='cuda'):
        self.dataloader = dataloader
        self.device = device

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for batch in self.dataloader:
            yield tuple(dict_to(x, self.device) for x in batch)