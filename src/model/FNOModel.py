import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import lightning as L

from .MLP import *
from .FourierLayer import *
from .Utilities import *
# from LossFunction import *


class ModelCallbacks(L.Callback):
    def on_train_start(self, trainer):
        # RichModelSummary(max_depth=1)
        pass
        
    def on_train_end(self, trainer):
        pass


class FNOModel(L.LightningModule):
    def __init__(self, num_layers, in_neurons, hidden_neurons, out_neurons, modesSpace, modesTime, input_size, learning_rate, restart_at_epoch_n, train_loader, loss_function):
        super().__init__()
        #self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.restart_at_epoch_n = restart_at_epoch_n
        #self.padding = time_padding # set padding here based on input_size
        self.n_batches = len(train_loader)
        self.n_training_samples = len(train_loader.dataset)
        self.loss_name = loss_function
        #train_batch, _ = next(iter(train_loader))
        #x_shape = train_batch.size()
        #self.register_buffer("meshgrid", get_meshgrid(x_shape))
        self.l = num_layers # number of layers
        # Network architechture
        self.p = nn.Linear(input_size, out_neurons)
        
        self.fourier = nn.ModuleList([FourierLayer(in_neurons, out_neurons, modesSpace, modesTime) for _ in range(self.l)])
        self.mlp = nn.ModuleList([MLP(out_neurons, hidden_neurons, out_neurons, kernel_size=1) for _ in range(self.l)])
        self.w = nn.ModuleList([nn.Conv3d(in_neurons, out_neurons, kernel_size=1) for _ in range(self.l)])
        
        #self.q = MLP(in_neurons, 4 * hidden_neurons, 1, kernel_size=1) 
        self.decoder = nn.Sequential(
            nn.Linear(in_neurons, 128),
            nn.ReLU(),
            nn.Linear(128, 4 * 601)  # 输出 4.0 个通道 * 601.0 个频点 = 2404.0 个标量
        )
        
        if loss_function == 'L2':
            self.loss_function = LpLoss()
        elif loss_function == 'MSE':
            self.loss_function = F.mse_loss
        elif loss_function == 'MAE':
            self.loss_function = F.l1_loss
    
            
    def forward(self, x): 
        #[B, 1, X, Y, Z] -> [B, X, Y, Z, 1]
        x = x.permute(0, 2, 3, 4, 1)
        # input dim: [B, X, Y, Z, C]
        meshgrid = get_meshgrid(x.shape).to(self.device)
        x = torch.concat((x, meshgrid), dim=-1) # [B, X, Y, Z, 3 + C]
        del meshgrid
        x = self.p(x) # [B, X, Y, Z, H]
        x = x.permute(0, 4, 1, 2, 3) # [B, H, X, Y, Z]
        #x = F.pad(x, [0, self.padding]) # Zero-pad
        for fourier_layer, mlp_layer, w_layer in zip(self.fourier, self.mlp, self.w):           
            x1 = fourier_layer(x)  
            x1 = mlp_layer(x1)    
            x2 = w_layer(x)     
            x = F.gelu(x1 + x2)
        
        #x = x[..., :-self.padding] # Unpad zeros
        '''
        x = self.q(x) # [B, 1, X, Y, Z]
        x = x.permute(0, 2, 3, 4, 1)  # [B, X, Y, Z, 1]
        x = x.squeeze_(dim=-1)
        return x
        '''
        # 形状变换：[Batch, 32, X, Y, Z] -> [Batch, 32.0]
        x = torch.mean(x, dim=[-3, -2, -1])
        # 形状变换：[Batch, 32.0] -> [Batch, 2404.0]
        x = self.decoder(x)
        # 4.0 代表 s11_real, s11_img, s21_real, s21_img 四个输出通道
        x = x.view(-1, 4, 601)
        
        return x
    

    def training_step(self, batch, batch_idx):
        x, y = batch 
        y_hat = self(x) # [B, X, Y, Z]
        train_loss = self.loss_function(y_hat, y) # .view(len(y), -1)
        train_mse = F.mse_loss(y_hat, y)
        log_dict = {'mse_loss': train_mse, 'train_' + self.loss_name + '_loss': train_loss}
        self.log_dict(log_dict, prog_bar=True, on_step=True, on_epoch=True)
        return train_loss 


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x) # [B,X,Y,Z]
        val_loss = self.loss_function(y_hat, y)
        self.log('val_' + self.loss_name + '_loss', val_loss, prog_bar=True, on_step=False, on_epoch=True)
        return val_loss
    
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x) # [B,X,Y,Z]
        
        test_loss = self.loss_function(y_hat, y)

        self.log('test_' + self.loss_name + '_loss', test_loss, prog_bar=True, on_step=False, on_epoch=True)
        
        return test_loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        del batch
        return self(x), y
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.restart_at_epoch_n)
        return optimizer