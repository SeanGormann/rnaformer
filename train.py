import sys
print(sys.executable)

from models import *
from data import *
from utils import *

nfolds = 20
device = 'cuda' if torch.cuda.is_available() else 'cpu'
bs = 128
num_workers = 4

df = pd.read_parquet('dataset/train_data.parquet')
#df  = pd.read_parquet('train_data_filtered.parquet')

#sequences = df['sequence'].values  # Assuming df is your DataFrame and it has a column named 'sequence'
#k = 3  # The length of the k-mers
#num_embeddings, kmers = count_unique_kmers(sequences, k)

seed_everything(2023)



exp1 = {
    "fname": "rnaformer-116",
    "fold": 15,
    "nfolds": 20,
    "s2n": 0.6,
    "perturb": True,
    "lr": 1e-3,
    "epochs": 100,
}
###

exp2 = {
    "fname": "rnaformer-111",
    "fold": 10,
    "nfolds": 20,
    "s2n": 0.6,
    "perturb": False,
    "lr": 1e-3,
    "epochs": 100,
}

exp3 = {
    "fname": "rnaformer-112",
    "fold": 10,
    "nfolds": 20,
    "s2n": 0.6,
    "perturb": False,
    "lr": 1e-3,
    "epochs": 40,
}
###

exp4 = {
    "fname": "rnaformer-111",
    "fold": 10,
    "nfolds": 20,
    "s2n": 0.6,
    "perturb": False,
    "lr": 1e-3,
    "epochs": 40,
}

exp5 = {
    "fname": "rnaformer-112",
    "fold": 10,
    "nfolds": 20,
    "s2n": 0.6,
    "perturb": False,
    "lr": 1e-3,
    "epochs": 40,
}

exp6 = {
    "fname": "rnaformer-113",
    "fold": 10,
    "nfolds": 20,
    "s2n": 0.6,
    "perturb": False,
    "lr": 1e-3,
    "epochs": 40,
}


###
"""
exp3 = {
    "fname": "rnaformer-98",
    "fold": 0,
    "nfolds": 5,
    "s2n": 0.55,
    "perturb": False,
    "lr": 1e-3,
    "epochs": 100,

}



## Do next
exp4 = {
    "fname": "rnaformer-99",
    "fold": 0,
    "nfolds": 5,
    "s2n": 0.55,
    "perturb": False,
    "lr": 1e-3,
    "epochs": 100,
    "zlossw": 1e-4,

}"""

cfg1 = {
    'dim': 192, # 300,
    'depth': 14,# 6,   32:1
    'headsize': 48,# 5,  32:1
    'ffdim': 4,
}


cfg2 = {
    'dim': 192, # 300,
    'depth': 18,# 6,   32:1
    'headsize': 48,# 5,  32:1
    'ffdim': 4,
} 


cfg3 = {
    'dim': 256, # 300,
    'depth': 12,# 6,   32:1
    'headsize': 64,# 5,  32:1
    'ffdim': 4,
}


cfg4 = {
    'dim': 192, # 300,
    'depth': 12,# 6,   32:1
    'headsize': 96,# 5,  32:1
    'ffdim': 4,
} 


cfg5 = {
    'dim': 160, # 300,
    'depth': 12,# 6,   32:1
    'headsize': 40,# 5,  32:1
    'ffdim': 4,
} 

cfg6 = {
    'dim': 160, # 300,
    'depth': 10,# 6,   32:1
    'headsize': 32,# 5,  32:1
    'ffdim': 4,
} 



#experiments = [exp1, exp2, exp3]

experiments = [(exp1, cfg1)] #, (exp2, cfg2), (exp3, cfg3)] #, (expp3, cfg3)]

#for fold in [0]: 
for exp, cfg in experiments: 
    for fold in [18]:  #fold exp["fold"]
        ds_train = RNA_Dataset(df, mode='train', fold=fold, nfolds=exp["nfolds"], perturb = exp["perturb"], s2n = exp["s2n"])
        ds_train_len = RNA_Dataset(df, mode='train', fold=fold,
                    nfolds=exp["nfolds"], mask_only=True, perturb = exp["perturb"], s2n = exp["s2n"])
        sampler_train = torch.utils.data.RandomSampler(ds_train_len)
        len_sampler_train = LenMatchBatchSampler(sampler_train, batch_size=bs,
                    drop_last=True)
        dl_train = DeviceDataLoader(torch.utils.data.DataLoader(ds_train,
                    batch_sampler=len_sampler_train, num_workers=num_workers,
                    persistent_workers=True), device)
    
        ds_val = RNA_Dataset(df, mode='eval', fold=fold, nfolds=exp["nfolds"], perturb = exp["perturb"], s2n = exp["s2n"])
        ds_val_len = RNA_Dataset(df, mode='eval', fold=fold, nfolds=exp["nfolds"],
                   mask_only=True, perturb = exp["perturb"], s2n = exp["s2n"])
        sampler_val = torch.utils.data.SequentialSampler(ds_val_len)
        len_sampler_val = LenMatchBatchSampler(sampler_val, batch_size=bs,
                   drop_last=False)
        dl_val= DeviceDataLoader(torch.utils.data.DataLoader(ds_val,
                   batch_sampler=len_sampler_val, num_workers=num_workers), device)
        gc.collect()
    
        print("Preparing the model...")
        data = DataLoaders(dl_train, dl_val)
    
        """if cfg == None:
            model = RNA_Model()
        else:
            model = RNAFormer(cfg)
            print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')"""
    
        dim = cfg["dim"]
        depth = cfg["depth"]
        hs = cfg["headsize"]
        ffdim = cfg["ffdim"]
        model = RNA_Model(dim=dim, depth=depth, head_size=hs, ffdim=ffdim)
            
        model = model.to(device)
        z_loss = AdjustedZLoss(z_loss_weight=1e-3)
    
        print("Starting training...")
        learn = Learner(
            data, model, loss_func=z_loss, 
            cbs=[GradientClip(3.0)],  # Added PrintLossCallback here
            metrics=[MAE()]
        ).to_fp16()
        
        OUT = "models/"
        #fname = "rnaformer-90"
        fname = exp["fname"]
        lr = exp["lr"]
        epochs = exp["epochs"]
        
        learn.fit_one_cycle(epochs, lr_max=lr, wd=0.05, pct_start=0.008)
        #learn.fit_one_cycle(320, lr_max=1e-3, wd=0.05, pct_start=0.002)
        
        print("Training finished. Saving the model...")
        model_path = os.path.join(OUT, f'{fname}_{fold}.pth')
        torch.save(learn.model.state_dict(), model_path)
        
        torch.cuda.empty_cache()
        gc.collect()
        
        # Analyze the validation dataset for sample lengths and average SNR
        print(f"Analyzing sample lengths and average SNR values for fold {fold}...")
        length_snr_map = {}
        for batch in dl_val:
            inputs, targets = batch
            lengths = inputs['mask'].sum(dim=1).cpu().numpy()
            sn_values = targets['sn'][:, 0].cpu().numpy()
    
            for length, sn in zip(lengths, sn_values):
                if length in length_snr_map:
                    length_snr_map[length].append(sn)
                else:
                    length_snr_map[length] = [sn]
    
        length_avg_snr_map = {length: sum(sn_list) / len(sn_list) for length, sn_list in length_snr_map.items()}

        # Define the absolute path to the folder to save fold stats
        current_directory = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
        stats_folder = os.path.join(current_directory, "stats_folder")
        
        # Ensure the stats folder exists
        if not os.path.exists(stats_folder):
            os.makedirs(stats_folder)

        print(f"Analyzing sample lengths and average SNR values for fold {fold}...")
        length_count_map, length_avg_snr_map = analyze_dataloader(dl_train)
        stats_file = save_fold_stats(length_count_map, length_avg_snr_map, fold, stats_folder)
    
        # Display the results
        for length, avg_snr in sorted(length_avg_snr_map.items()):
            print(f"Fold {fold}, Length: {length}, Count: {length_count_map[length]}, Average SNR: {avg_snr:.2f}")
        
            
        # Push model and stats to GitHub
        print(f"Pushing the model and stats for fold {fold} to GitHub...")
        push_to_github(model_path, f"Add trained model for fold {fold}")
        push_to_github(stats_file, f"Add fold {fold} stats")
    
        torch.cuda.empty_cache()
        gc.collect()
    
        print(f"Fold {fold} processing done!\n")






"""
class RNA_Trainer:
    def __init__(self, config):
        self.config = config
        self.model = None  # Initialize your model here
        self.train_loader = None
        self.val_loader = None
    
    def load_data(self, file_path):
        # Implement loading and preprocessing of your data
        # Set self.train_loader and self.val_loader
        
    def train(self):
        # Implement the training logic here, use self.config to get the configuration parameters
        
    def validate(self):
        # Implement the validation logic here
        
    def test(self, test_data):
        # Implement the testing logic here
    
    def save_model(self, file_path):
        # Implement logic to save the model

    def load_model(self, file_path):
        # Implement logic to load a pre-trained model
"""