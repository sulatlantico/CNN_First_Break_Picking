from monai.losses import (DiceCELoss, GeneralizedDiceFocalLoss, FocalLoss, DiceLoss)
from torchmetrics.classification import (BinaryF1Score, BinaryPrecision,
                                         BinaryAccuracy,BinaryRecall)
from pytorch_lightning import LightningModule
from monai.networks.nets import UNet
from torch.nn import Sequential, Sigmoid, CrossEntropyLoss, Softmax, Module
from torch.optim import Adam
from torch import device

class Unet(LightningModule): 
    def __init__(self): 
        super(Unet,self).__init__()
        self.model = Sequential(UNet(spatial_dims=1,in_channels=1,
                              out_channels=1,channels=(16,32,64,128,256), #8,16,32,64,128
                               strides=(1,1,1,1), num_res_units=2,
                                  kernel_size=7, dropout=.2, act ='PRELU'), #act = 'TANH'
                              Sigmoid()) 
        self.loss = DiceCELoss() #DiceCELoss()
        self.train_acc = BinaryAccuracy()
        self.valid_acc = BinaryAccuracy()
        self.f1_score = BinaryF1Score()
        self.prec = BinaryPrecision()
        self.recall = BinaryRecall()

    def forward(self,x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(),lr=0.005) #0.005
        return optimizer
        
    def training_step(self,train_batch,batch_idx):
        x, y = train_batch['x'], train_batch['y'] 
        #forward pass
        z = self.model(x)
        loss = self.loss(z, y)
        t_acc = self.train_acc(z,y)
        f1_score = self.f1_score(z,y)
        precision = self.prec(z,y)
        recall = self.recall(z,y)
        self.log('Train_loss',loss, on_step=False, on_epoch = True, prog_bar = True,
                enable_graph = True)
        self.log('Train_acc', t_acc, on_step=False, on_epoch = True, prog_bar = True)
        self.log('F1_Score', f1_score, on_step=False, on_epoch = True, prog_bar = True)
        self.log('Precision', precision, on_step=False, on_epoch = True, prog_bar = True)
        self.log('Recall', recall, on_step=False, on_epoch = True, prog_bar = True)
        return loss
    
    def validation_step(self,val_batch,batch_idx):
        x, y = val_batch["x"], val_batch["y"]
        #forward pass
        z = self.model(x)
        loss = self.loss(z, y)
        v_acc = self.valid_acc(z,y)
        val_f1_score = self.f1_score(z,y)
        val_precision = self.prec(z,y)
        val_recall = self.recall(z,y)
        self.log('Val_loss',loss, on_step=False, on_epoch = True, prog_bar = True)
        self.log('Valid_acc', v_acc, on_step=False, on_epoch = True, prog_bar = True)
        self.log('Val_F1_Score', val_f1_score, on_step=False, on_epoch = True, prog_bar = True)
        self.log('Val_Precision', val_precision, on_step=False, on_epoch = True, prog_bar = True)
        self.log('Val_Recall', val_recall, on_step=False, on_epoch = True, prog_bar = True)
        return loss
     
    def predict_step(self,batch, batch_idx,dataloader_idx=0): 
        return self(batch)  
    

if __name__ == '__main__':
    model = Unet()


