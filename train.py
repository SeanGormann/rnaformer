import sys
print(sys.executable)

from models import *
from data import *
from utils import *

nfolds = 20
device = 'cuda' if torch.cuda.is_available() else 'cpu'
bs = 128
num_workers = 2

df = pd.read_parquet('dataset/train_data.parquet')
#df  = pd.read_parquet('train_data_filtered.parquet')

#sequences = df['sequence'].values  # Assuming df is your DataFrame and it has a column named 'sequence'
#k = 3  # The length of the k-mers
#num_embeddings, kmers = count_unique_kmers(sequences, k)

seed_everything(2023)

exp1 = {
    "fname": "rnaformer-96",
    "fold": 5,
    "nfolds": 20,
    "s2n": 0.6,
    "perturb": True,
    "lr": 1e-3,
    "epochs": 100,
}

exp2 = {
    "fname": "rnaformer-97",
    "fold": 8,
    "nfolds": 10,
    "s2n": 0.6,
    "perturb": False,
    "lr": 1e-3,
    "epochs": 110,

}


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

}

cfg1 = {
    'rna_model_dim': 192, # 300,
    'rna_model_num_heads': 6,# 6,   32:1
    'rna_model_num_encoder_layers': 12,# 5,  32:1
    'rna_model_num_lstm_layers': 0,
    'rna_model_lstm_dropout': 0, #0.1,
    'rna_model_first_dropout': 0.1,
    'rna_model_encoder_dropout': 0.1,
    'rna_model_mha_dropout': 0,
    'rna_model_ffn_multiplier': 5,
}

experiments = [exp1, exp2, exp3]

#for fold in [0]: 
for exp in experiments: 
    ds_train = RNA_Dataset(df, mode='train', fold=exp["fold"], nfolds=exp["nfolds"], perturb = exp["perturb"], s2n = exp["s2n"])
    ds_train_len = RNA_Dataset(df, mode='train', fold=exp["fold"],
                nfolds=exp["nfolds"], mask_only=True, perturb = exp["perturb"], s2n = exp["s2n"])
    sampler_train = torch.utils.data.RandomSampler(ds_train_len)
    len_sampler_train = LenMatchBatchSampler(sampler_train, batch_size=bs,
                drop_last=True)
    dl_train = DeviceDataLoader(torch.utils.data.DataLoader(ds_train,
                batch_sampler=len_sampler_train, num_workers=num_workers,
                persistent_workers=True), device)

    ds_val = RNA_Dataset(df, mode='eval', fold=exp["fold"], nfolds=exp["nfolds"], perturb = exp["perturb"], s2n = exp["s2n"])
    ds_val_len = RNA_Dataset(df, mode='eval', fold=exp["fold"], nfolds=exp["nfolds"],
               mask_only=True, perturb = exp["perturb"], s2n = exp["s2n"])
    sampler_val = torch.utils.data.SequentialSampler(ds_val_len)
    len_sampler_val = LenMatchBatchSampler(sampler_val, batch_size=bs,
               drop_last=False)
    dl_val= DeviceDataLoader(torch.utils.data.DataLoader(ds_val,
               batch_sampler=len_sampler_val, num_workers=num_workers), device)
    gc.collect()

    print("Preparing the model...")
    data = DataLoaders(dl_train, dl_val)
    model = RNA_Model()
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
    model_path = os.path.join(OUT, f'{fname}.pth')
    torch.save(learn.model.state_dict(), model_path)
    
    torch.cuda.empty_cache()
    gc.collect()
    
    print("Pushing the model to GitHub...")
    commit_message = f"Add trained model {fname}"
    push_to_github(model_path, commit_message)
    
    print("Done!")






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