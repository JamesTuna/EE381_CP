import pandas as pd
import numpy as np
import time, random
from sklearn.preprocessing import RobustScaler, normalize
from sklearn.model_selection import KFold

from torch.utils.data import DataLoader
from utils.myModel import *
from utils.loader import *
from pytorch_ranger import Ranger
from torch.utils.tensorboard import SummaryWriter

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--n_Tlayer",type=int,default=4)
parser.add_argument("--n_head",type=int,default=4)
parser.add_argument("--n_Elayer",type=int,default=2)
parser.add_argument("--Etype",type=str,default='GRU')
parser.add_argument("--use_conv",action="store_true")
parser.add_argument("--dropout",type=float,default=0)
args = parser.parse_args()

trial_name = f"T{args.n_Tlayer}h{args.n_head}{args.Etype}{args.n_Elayer}"
trial_name += "conv" if args.use_conv else ""
trial_name += f"_dropout{args.dropout:.2f}"

print("init trial ",trial_name)

print("Loading data...")
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
masks=np.array(train['u_out']==0).reshape(-1, 80) 
targets = train[['pressure']].to_numpy().reshape(-1, 80)

print("One hot encoding...")
for dset in ('train','test'):
    df = eval(dset)
    df['R'] = df['R'].astype(str)
    df['C'] = df['C'].astype(str)
    df = pd.get_dummies(df)
    exec(f'{dset}=df')


print("Dropping id and labels...")
train.drop(['pressure', 'id', 'breath_id'], axis=1, inplace=True)
test = test.drop(['id', 'breath_id'], axis=1)

print("Normalizing...")
RS = RobustScaler()
train = RS.fit_transform(train)
test = RS.transform(test)

print("Reshaping...")
train = train.reshape(-1, 80, train.shape[-1])
test = test.reshape(-1, 80, train.shape[-1])

print('train:',train.shape)
print('test:',test.shape)

kf = KFold(n_splits=10,shuffle=True,random_state=0) # forbid random_state, make some randomness between models in different runs

train_features=[train[i] for i in list(kf.split(train))[0][0]]
val_features=[train[i] for i in list(kf.split(train))[0][1]]
train_targets=[targets[i] for i in list(kf.split(targets))[0][0]]
val_targets=[targets[i] for i in list(kf.split(targets))[0][1]]
train_masks=[masks[i] for i in list(kf.split(targets))[0][0]]
val_masks=[masks[i] for i in list(kf.split(targets))[0][1]]

print(f"{len(train_features):5d} samples to train")
print(f"{len(val_features):5d} samples to validate")


train_dataset = TrainDataset(train_features,train_targets,train_masks)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
del train_features

val_dataset = TrainDataset(val_features,val_targets,val_masks)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
del val_features

# model init
print("init model...")
model = MyModel(in_dim = train.shape[-1], out_dim = 1,embd_dim=128,
                n_transformer_layers=args.n_Tlayer, nheads = args.n_head, dropout=args.dropout,
                n_rnn_layers=args.n_Elayer, rnn_type = args.Etype,use_conv = args.use_conv).cuda()
optimizer = Ranger(model.parameters(), lr=8e-5)
criterion = nn.L1Loss(reduction='none')
epochs=150
val_metric = 100
best_metric = 100
cos_epoch=int(epochs*0.75)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,(epochs-cos_epoch)*len(train_dataloader))
steps_per_epoch=len(train_dataloader)
val_steps=len(val_dataloader)

writer = SummaryWriter(comment=trial_name)

# train loop
print("start training...")
for epoch in range(epochs):
    model.train()
    train_loss=0
    t=time.time()
    for step,batch in enumerate(train_dataloader):
        #series=batch.to(device)#.float()
        features,targets,mask=batch
        features=features.cuda()
        targets=targets.cuda()
        mask=mask.cuda()
        #exit()

        optimizer.zero_grad()
        output=model(features)
        #exit()
        #exit()

        loss=criterion(output,targets)#*loss_weight_vector
        loss=torch.masked_select(loss,mask)
        loss=loss.mean()
        loss.backward()
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()
        optimizer.step()

        train_loss+=loss.item()
        #scheduler.step()
        if step % 100 == 0:
            
            print (f"Epoch [{epoch+1}/{150}] Step [{step+1}/{steps_per_epoch}] Loss: {train_loss/(step+1):.3f} Time: {time.time()-t:.1f}",
                   end='\r',flush=True)
        if epoch > cos_epoch:
            scheduler.step()
        #break
    print('')
    train_loss/=(step+1)
    
    writer.add_scalar('Loss/train', train_loss, epoch)

    #exit()
    model.eval()
    val_metric=[]
    val_loss=0
    t=time.time()
    preds=[]
    truths=[]
    masks=[]
    for step,batch in enumerate(val_dataloader):
        features,targets,mask=batch
        features=features.cuda()
        targets=targets.cuda()
        mask=mask.cuda()
        with torch.no_grad():
            output=model(features)

            loss=criterion(output,targets)
            loss=torch.masked_select(loss,mask)
            loss=loss.mean()
            val_loss+=loss.item()
            preds.append(output.cpu())
            truths.append(targets.cpu())
            masks.append(mask.cpu())
    
    preds=torch.cat(preds).numpy()
    truths=torch.cat(truths).numpy()
    masks=torch.cat(masks).numpy()
    val_metric=(np.abs(truths-preds)*masks).sum()/masks.sum()#*stds['pressure']
    
    val_loss/=(step+1)
    writer.add_scalar('Loss/Validation', val_loss, epoch)
    
    print(f'validation loss:{val_loss:.4f}')
    print('')


    if val_metric < best_metric:
        best_metric=val_metric
        torch.save(model.state_dict(),trial_name+'.pt')

