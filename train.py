import sys
print(sys.executable)

from models import *
from data import *
from utils import *

nfolds = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
bs = 128
num_workers = 2

#df = pd.read_parquet('dataset/train_data.parquet')
df  = pd.read_parquet('train_data_filtered.parquet')

#sequences = df['sequence'].values  # Assuming df is your DataFrame and it has a column named 'sequence'
#k = 3  # The length of the k-mers
#num_embeddings, kmers = count_unique_kmers(sequences, k)



for fold in [0]: # running multiple folds at kaggle may cause OOM
    ds_train = RNA_Dataset(df, mode='train', fold=fold, nfolds=nfolds)
    ds_train_len = RNA_Dataset(df, mode='train', fold=fold,
                nfolds=nfolds, mask_only=True)
    sampler_train = torch.utils.data.RandomSampler(ds_train_len)
    len_sampler_train = LenMatchBatchSampler(sampler_train, batch_size=bs,
                drop_last=True)
    dl_train = DeviceDataLoader(torch.utils.data.DataLoader(ds_train,
                batch_sampler=len_sampler_train, num_workers=num_workers,
                persistent_workers=True), device)

    ds_val = RNA_Dataset(df, mode='eval', fold=fold, nfolds=nfolds)
    ds_val_len = RNA_Dataset(df, mode='eval', fold=fold, nfolds=nfolds,
               mask_only=True)
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
    
    class PrintLossCallback(Callback):
        def after_epoch(self):
            print(f"Epoch {self.epoch}: Train loss: {self.learn.recorder.losses[-1]:.4f}, Valid loss: {self.learn.recorder.values[-1][0]:.4f}")

    print("Starting training...")
    learn = Learner(
        data, model, loss_func=adjusted_loss, 
        cbs=[GradientClip(3.0), PrintLossCallback()],  # Added PrintLossCallback here
        metrics=[MAE()]
    ).to_fp16()
    
    OUT = "./models/"
    fname = "rnaformer-66"
    
    learn.fit_one_cycle(45, lr_max=45e-5, wd=0.05, pct_start=0.02)
    
    print("Training finished. Saving the model...")
    model_path = os.path.join(OUT, f'{fname}_{fold}.pth')
    torch.save(learn.model.state_dict(), model_path)
    
    torch.cuda.empty_cache()
    gc.collect()
    
    print("Pushing the model to GitHub...")
    commit_message = f"Add trained model {fname}_{fold}"
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