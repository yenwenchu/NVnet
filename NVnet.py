#ensure that PyTorch 1.2.0 and torch-geometric 1.3.2 are installed
import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool,GINConv,GCNConv,SAGEConv
from torch_geometric.datasets import TUDataset
from torch.nn import Sequential, Linear,Parameter,ReLU
import torch.optim as optim
from torch_geometric.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from torch.nn.modules import Module
from torch_scatter import scatter_add,scatter_max
import numpy as np
from torch_geometric.transforms import OneHotDegree
import torch_geometric.data
from torch_geometric.utils import remove_self_loops,add_self_loops

EPS = 1.0e-4


class ConvsBlock(Module):
    def __init__(self, HP,cldim):
        super(ConvsBlock, self).__init__()
        self.HP=HP
        self.ConvsList=torch.nn.ModuleList([HP['gnn'](**HP['HP_before_last_layer']) for i in range(HP['layers']-1)])
        self.BNList=torch.nn.ModuleList([torch.nn.BatchNorm1d(cldim)for i in range(HP['layers']-1)])
        self.ConvsList.append(HP['gnn'](**HP['HP_last_layer']))
        self.BNList.append(torch.nn.BatchNorm1d(cldim))
    def forward(self, x,edge_index):
        hlist=[]
        h=x
        for i in range(self.HP['layers']-1):
            h = self.BNList[i](h)
            h=self.HP['act_bll'](self.ConvsList[i](h,edge_index))
            hlist.append(h)
        h = self.BNList[self.HP['layers'] - 1](h)
        h = self.HP['act_ll'](self.ConvsList[self.HP['layers']-1](h, edge_index))
        hlist.append(h)
        x=torch.cat(hlist,dim=-1)
        return x

class ReconBlock(Module):
    def __init__(self):
        super(ReconBlock, self).__init__()
    def forward(self, x, pos_edge_index,neg_edge_index,batch):
        Lrc=self.recon_loss(x,pos_edge_index,neg_edge_index,batch)
        return Lrc
    def recon_loss(self, x, pos_edge_index,neg_edge_index,batch):
        pos_loss = global_mean_pool(-torch.log(EPS+self.decoder(x, pos_edge_index, sigmoid=True)),batch[pos_edge_index[0]])
        pos_loss = torch.sum(pos_loss)
        if len(neg_edge_index[0])>0:
            neg_loss = global_mean_pool(-torch.log(EPS+1 - self.decoder(x, neg_edge_index, sigmoid=True)),batch[neg_edge_index[0]])
            neg_loss = torch.sum(neg_loss)
            Lreg=pos_loss + neg_loss
        else:
            Lreg= pos_loss
        return Lreg
    def decoder(self, x, edge_index, sigmoid=True):
        value = (x[edge_index[0]] * x[edge_index[1]]).sum(dim=1)

        return torch.sigmoid(value) if sigmoid else value

class ClfBlock(Module):
    def __init__(self,HP,num_classes):
        super(ClfBlock, self).__init__()
        self.HP=HP
        # n*(heads*hidden_channels)
        self.Features_For_All_Head = Linear(HP['att_in_channels'], HP['att_hidden_channels']*HP['heads'], bias=False)
        self.FC1 = Linear(HP['att_hidden_channels']*HP['heads'], HP['att_hidden_channels'])
        self.FC2 = Linear(HP['att_hidden_channels'], num_classes)
        self.phi=Parameter(torch.nn.init.xavier_uniform_(torch.zeros((HP['heads'], 2*HP['att_hidden_channels'],1))))
    def forward(self, x,batch,mask):
        midx=torch.stack(list(torch.where(mask==1))).to(torch.long) #2 , medges_num
        x = self.Features_For_All_Head(x).reshape(-1,self.HP['heads'],self.HP['att_hidden_channels'])  # nodes_num, head,att_hidden_channelscp
        cx=torch.cat([x[midx[0,:]],x[midx[1,:]]],dim=-1).permute(1,0,2 )  #head, num_edges, 2*att_hidden_channels
        weights_pre=F.leaky_relu(cx.matmul(self.phi).permute(1,0,2).reshape(-1,self.HP['heads']*1))#medge_mum  heads*1
        weights=torch.sigmoid(weights_pre)#medge_mum  heads*1
        x=x[midx[1,:]]*weights.unsqueeze(-1)  #medge_num, heads, hidden_channel
        x=x.reshape(-1,self.HP['heads']*self.HP['att_hidden_channels'])  #medge_num, heads*hidden_channel
        x=scatter_add(x,midx[0,:],dim=0) #node_nums ,heads*hidden_channel
        x=self.FC2( F.dropout( F.leaky_relu(self.FC1(x)),p=0.5, training=self.training))
        tmp=torch.exp(x-torch.max(x,dim=-1,keepdim=True)[0])+EPS
        preds = tmp/torch.sum(tmp,dim=-1,keepdim=True)
        yp = global_mean_pool(preds,batch)
        return torch.log(yp),preds

class NVnet(Module):
    def __init__(self,HP,num_features,num_classes,cldim):
        super(NVnet, self).__init__()
        self.HP=HP
        self.covb= ConvsBlock(HP['CBHP'],cldim)
        self.recb= ReconBlock()
        self.clfb= ClfBlock(HP['CFBHP'],num_classes)
        self.fc=Linear(num_features,cldim)
        self.num_classes=num_classes

    def forward(self,x,edge_index,neg_edge_index,batch,mask):
        x=self.fc(x)
        x=self.covb(x,edge_index)
        Lrc=self.recb(x,edge_index,neg_edge_index,batch)/(torch.max(batch)+1)
        yp,preds=self.clfb(x,batch,mask)
        return yp,Lrc,preds


def get_mask_and_edge_index(batch,pos_edge_index):
    nodes_num=len(batch)
    mask=torch.zeros((nodes_num,nodes_num)).to('cuda')
    idx = torch.stack(list(torch.where(mask == 0)))
    r = batch[idx[0, :]] == batch[idx[1, :]]
    mask[idx[0,r],idx[1,r]]=1
    m = batch[idx[0, :]] != batch[idx[1, :]]
    nmask=torch.ones((nodes_num,nodes_num)).to('cuda')
    nmask[idx[0,m],idx[1,m]]=0
    nmask[pos_edge_index[0,:], pos_edge_index[1,:]] = 0
    neg_edge_index=torch.stack(list(torch.where(nmask==1)))
    return mask,neg_edge_index

def one_epoch_train(net,optimizer,data_loader,num_samples,device):
    net.train()
    loss_epoch=0.0
    for data in data_loader:
        if data.num_graphs==1:
            ds=[]
            for i in range(2):
                ds.append(torch_geometric.data.Data(x=data.x,edge_index=data.edge_index,y=data.y))
            data=iter(DataLoader(ds, batch_size=2, shuffle=False)).next()
            optimizer.zero_grad()
            data.to(device)
            edge_index, _ = remove_self_loops(data.edge_index)
            edge_index, _ = add_self_loops(edge_index)
            mask, neg_edge_index = get_mask_and_edge_index(data.batch, edge_index)
            output, Lrc,preds = net(data.x, edge_index, neg_edge_index, data.batch, mask)
            Lc= F.nll_loss(output, data.y)
            Lreg = F.nll_loss(torch.log(preds), data.y[data.batch])
            loss =   Lc+Lreg*0.2+0.04*Lrc
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()*data.num_graphs
            #print('single graph train')
        else:
            optimizer.zero_grad()
            data.to(device)
            edge_index, _ = remove_self_loops(data.edge_index)
            edge_index, _ = add_self_loops(edge_index)
            mask,neg_edge_index=get_mask_and_edge_index(data.batch, edge_index)
            output,Lrc,preds= net(data.x, edge_index, neg_edge_index,data.batch,mask)
            Lc = F.nll_loss(output, data.y)
            Lreg= F.nll_loss(torch.log(preds),data.y[data.batch])
            loss = Lc + Lreg * 0.2+0.04*Lrc
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item() *data.num_graphs
    return loss_epoch/num_samples

def test(net,data_loader,num_samples,device):
    net.eval()
    acc=0.0
    with torch.no_grad():
        for data in data_loader:
            if data.num_graphs == 1:
                ds = []
                for i in range(2):
                    ds.append(torch_geometric.data.Data(x=data.x, edge_index=data.edge_index, y=data.y))
                data = iter(DataLoader(ds, batch_size=2, shuffle=False)).next()
                data = data.to(device)
                edge_index, _ = remove_self_loops(data.edge_index)
                edge_index, _ = add_self_loops(edge_index)
                mask, neg_edge_index = get_mask_and_edge_index(data.batch, edge_index)
                output, _, __ = net(data.x, edge_index, neg_edge_index, data.batch, mask)
                _, pre = output.topk(1)
                pre = pre.squeeze(-1)
                acc += float(pre.eq(data.y).sum().item())/2
                print('single graph test')
            else:
                data = data.to(device)
                edge_index, _ = remove_self_loops(data.edge_index)
                edge_index, _ = add_self_loops(edge_index)
                mask, neg_edge_index = get_mask_and_edge_index(data.batch, edge_index)
                output, _,__= net(data.x, edge_index, neg_edge_index, data.batch, mask)
                _, pre = output.topk(1)
                pre = pre.squeeze(-1)
                acc += float(pre.eq(data.y).sum().item())
    return acc/num_samples


def main_process(seed, dataname, HP,transform_flag,cldim,LR = 0.001,EPOCHS = 100,BATCHSIZE = 20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if transform_flag==True:
        dataset = TUDataset(root='data/' + dataname, name=dataname ,transform=OneHotDegree(1000))
    else:
        dataset = TUDataset(root='data/' + dataname, name=dataname)

    skf = StratifiedKFold(n_splits=10, shuffle=True,random_state=seed)
    acc_rate_list = []
    for train_index, test_index in skf.split(dataset, dataset.data.y):
        train_set = [dataset[int(index)] for index in train_index]
        test_set = [dataset[int(index)] for index in test_index]

        train_data_loader = DataLoader(train_set, batch_size=BATCHSIZE, shuffle=True)
        test_data_loader = DataLoader(test_set, batch_size=BATCHSIZE)
        net = NVnet(HP,dataset.num_features,dataset.num_classes,cldim).to(device)
        optimizer = optim.Adam(net.parameters(), lr=LR,weight_decay=0)
        lr_schedule = optim.lr_scheduler.StepLR(optimizer, 20, 0.5)
        for epoch in range(EPOCHS):
            loss_epoch = one_epoch_train(net, optimizer,train_data_loader,len(train_set), device)
            lr_schedule.step()
        acc_rate = test(net, test_data_loader, len(test_set),device)
        print(acc_rate)
        acc_rate_list.append(acc_rate)
    acc_records = np.array(acc_rate_list)
    print(dataname)
    print('acc records', acc_records)
    print(dataname+" acc mean:{}".format(acc_records.mean()))
    print(dataname+" acc std:{}".format(acc_records.std()))

CL_nn_dim=16
GnnHP_b_ll = dict({'nn':Sequential(Linear(CL_nn_dim, CL_nn_dim), ReLU(), Linear(CL_nn_dim, CL_nn_dim)),'train_eps':True})
GnnHP_ll = dict({'nn':Sequential(Linear(CL_nn_dim, CL_nn_dim), ReLU(), Linear(CL_nn_dim, CL_nn_dim)),'train_eps':True})
ConvsBlockHP = dict({'gnn': GINConv,
                     'HP_before_last_layer': GnnHP_b_ll,
                     'HP_last_layer': GnnHP_ll,
                     'layers': 3,
                     'act_bll':F.relu,
                     'act_ll':F.relu})
ClfBlockHP=dict({'heads':4,
                 'att_in_channels':ConvsBlockHP['layers'] * CL_nn_dim,
                 'att_hidden_channels':CL_nn_dim})
NVnetHP=dict({  'CBHP':ConvsBlockHP,
                'CFBHP':ClfBlockHP})
main_process(0,'IMDB-BINARY',NVnetHP,True,CL_nn_dim)


CL_nn_dim=32
GnnHP_b_ll = dict({'in_channels':CL_nn_dim ,'out_channels':CL_nn_dim })
GnnHP_ll = dict({'in_channels':CL_nn_dim ,'out_channels':CL_nn_dim })
ConvsBlockHP = dict({'gnn': GCNConv,
                     'HP_before_last_layer': GnnHP_b_ll,
                     'HP_last_layer': GnnHP_ll,
                     'layers': 3,
                     'act_bll':F.relu,
                     'act_ll':F.relu})

ClfBlockHP=dict({'heads':4,
                 'att_in_channels':ConvsBlockHP['layers'] * CL_nn_dim,
                 'att_hidden_channels':CL_nn_dim})
NVnetHP=dict({  'CBHP':ConvsBlockHP,
                'CFBHP':ClfBlockHP})
main_process(0,'COLLAB',NVnetHP,True,CL_nn_dim)


#main_process(0,'MUTAG',NVnetHP,False,CL_nn_dim)
#main_process(0,'PROTEINS',NVnetHP,False,CL_nn_dim)

