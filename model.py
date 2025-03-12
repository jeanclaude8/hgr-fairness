import lightning as L
from torch import nn
from maxcorr import indicator
from torch import optim 
from torch.optim.lr_scheduler import ReduceLROnPlateau

class DecorellModel(nn.Module):
    def __init__(self, input_shape, hidden_shape, output_shape):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64,hidden_shape)

        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_shape, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, output_shape) 
        )
    
    def forward(self, x):
        z = self.encoder(x)  
        transformed_x = self.decoder(z)  
        return transformed_x
    
class DecorellModelTraining(L.LightningModule):
    def __init__(self, model: nn.Module, lr : float, weight_decay: float, alpha : float, beta : float, epsilon : float, costum_loss: callable): 
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.costum_loss = costum_loss
        #self.gradient_stats = {}
        #self.save_hyperparameters() 
    
    def training_step(self, batch, batch_idx):
        z, x, _ = batch
        x_hat = self.model(x)
        lagrangian, constraint = self.costum_loss(x, x_hat, z, self.alpha, self.beta, self.epsilon)
        self.log('lagrangian_loss', lagrangian, on_step=False, on_epoch=True, prog_bar=True, logger=True) 
        self.log('constraint', constraint, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return lagrangian
    
    def validation_step(self, batch, batch_idx):
        ind = indicator('hgr', 'sk', 'torch')
        z, x, _ = batch
        x_hat = self.model(x)
        lagrangian, constraint = self.costum_loss(x, x_hat, z, self.alpha, self.beta, self.epsilon)
        correlation = ind.compute(x, x_hat)
        fairness = ind.compute(z, x_hat)
        self.log('lagrangian_loss_val', lagrangian, on_step=False, on_epoch=True, prog_bar=True, logger=True) 
        self.log('constraint_val', constraint, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('correlation_val', correlation, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('fairness_val', fairness, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return lagrangian


    def predict_step(self, batch, batch_idx):
        _, x, _ = batch
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5),
            'monitor': 'lagrangian_loss',  
            'interval': 'epoch'         
        }
        return {'optimizer' : optimizer, 'lr_scheduler' : scheduler}
    
