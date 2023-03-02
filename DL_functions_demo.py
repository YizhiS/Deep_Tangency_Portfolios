# copy right @ Feng, Jiang, Li and Song "Deep Tangency Portfolios" (2023)

import numpy as np
import pandas as pd
import random
import torch
import torch.nn.functional as F     
from torch.nn import init
def cal_portfolio_vw_return(df):
    temp = df[['excess_ret', 'size_value']].dropna()
    return (temp['excess_ret'] * temp['size_value']).sum() / temp['size_value'].sum()


class Net_SR(torch.nn.Module):  
    def __init__(self,data_input):
#         n_timelen,n_firm,stock_return,g_bench, n_feature, n_hidden,n_hidden_2,
#                  n_hidden_3,n_hidden_4,n_output
        super(Net_SR, self).__init__()     
        self.bond_ret = data_input['bond_return']
        self.bond_ret_oos = data_input['bond_return_oos']
        self.g_bench = data_input['factor']
        self.layers = data_input['layers']
        print(self.g_bench.shape)
        n_firm = data_input['num_firm']
        n_feature = data_input['num_feature']
        n_bench = self.g_bench.shape[1]
        n_output = 1
        if len(self.layers)>= 1:
            
            self.hidden = torch.nn.Linear(n_feature, self.layers[0])   
#             self.batchnorm_1 = torch.nn.BatchNorm1d(n_firm , affine=False)
        if len(self.layers)>= 2:
            self.hidden_2 = torch.nn.Linear(self.layers[0], self.layers[1])
#             self.batchnorm_2 = torch.nn.BatchNorm1d(n_firm , affine=False)
        if len(self.layers)>= 3:
            self.hidden_3 = torch.nn.Linear(self.layers[1], self.layers[2])
#             self.batchnorm_3 = torch.nn.BatchNorm1d(n_firm , affine=False)
        if len(self.layers)>= 4:
            self.hidden_4 = torch.nn.Linear(self.layers[2], self.layers[3])
#             self.batchnorm_4 = torch.nn.BatchNorm1d(n_firm , affine=False)
        self.output = torch.nn.Linear(self.layers[-1], 1)   
        
        self.batchnorm = torch.nn.BatchNorm1d(n_firm , affine=False)
#         self.activate = torch.nn.Tanh()
        
#         self.w_allocate = torch.nn.Linear(n_output+n_bench, 1, bias=False)
#         self.regress = torch.nn.Linear(n_output+n_bench,600)

    
    def forward(self, x,sample_period=False,print_state =False):  

        drop_ = torch.nn.Dropout(p=0)

        activate = torch.nn.Tanh()
        if len(self.layers)>= 1:
            x = (1*activate(self.hidden(x)))
        if len(self.layers)>= 2:
            x = (1*activate(self.hidden_2(x)))
        if len(self.layers)>= 3:
            x = (1*activate(self.hidden_3(x)))
        if len(self.layers)>= 4:
            x = (1*activate(self.hidden_4(x)))
            
        mid_x = x.clone().detach()

        
        x = self.output(drop_(x))            # output deep characteristics
#         print(x.shape)
        x = self.batchnorm(x)
        
        transformed_x_a = -50*torch.exp(-8*x)
        transformed_x_b = -50*torch.exp(8*x)
        
        w_ = F.softmax(transformed_x_a,dim=1) - F.softmax(transformed_x_b,dim=1)
        w_ = w_.reshape([w_.shape[0],w_.shape[1]])
        
        if sample_period =='oos':
            f_ = (w_*self.bond_ret_oos).sum(axis=1).reshape(-1,1)
            f_ = f_*self.sign
            return f_,w_,x.reshape(x.shape[0],x.shape[1]),0
        
        else:
            f_ = (w_*self.bond_ret).sum(axis=1).reshape(-1,1)
            
            self.sign = torch.sign(f_.mean())
            f_ = f_*self.sign
            
        F_ = torch.cat([self.g_bench,f_],axis=1)  
        
#         print(F_.T.cov())
        w_allocate = torch.inverse(F_.T.cov())@F_.mean(axis=0)
#         print(w_allocate.data.numpy(),(f_.mean()/f_.std()*3.464).data.numpy())
#         OP = self.w_allocate(F_)
# #         print(OP.shape)
#         E = F_.mean(axis=0)
#         Cov = F_.T.cov()
#         loss = torch.exp(- E/Std )
        loss = torch.exp(- F_.mean(axis=0)@torch.inverse(F_.T.cov())@F_.mean(axis=0))
#         print(torch.sqrt(-torch.log(loss.data)*12))
#         loss = torch.exp(-torch.matmul(w_allocate,E) + 4/2*torch.matmul(torch.matmul(w_allocate,Cov),w_allocate))
#         if (-loss.data)
        if print_state == True:
#             print(E.data.numpy(),Std.data.numpy())
            print(-torch.log(loss.data))
            
        return loss,f_,w_,x.reshape(x.shape[0],x.shape[1]),mid_x
#         return loss,f_,w_,x.reshape(x.shape[0],x.shape[1])
    


def penalty_adding(net,list_layer):
    if len(list_layer)>=1:
        penalty = torch.norm(net.hidden.weight,p=1) -  torch.norm(torch.diag(net.hidden.weight),p=1) 
        num_node = 132*list_layer[0]
    if len(list_layer)>=2:
        penalty = penalty + torch.norm(net.hidden_2.weight,p=1) -  torch.norm(torch.diag(net.hidden_2.weight),p=1) 
        num_node = num_node+list_layer[0]*list_layer[1]
    if len(list_layer)>=3:
        penalty = penalty + torch.norm(net.hidden_3.weight,p=1) -torch.norm(torch.diag(net.hidden_3.weight),p=1) 
        num_node = num_node+list_layer[1]*list_layer[2]
    if len(list_layer)>=4:
        penalty = penalty + torch.norm(net.hidden_4.weight,p=1) -torch.norm(torch.diag(net.hidden_4.weight),p=1) 
        num_node = num_node+list_layer[2]*list_layer[3]
    num_node = num_node + list_layer[-1]
    penalty = (penalty )
    return penalty





def initialization_(net,list_layer):
    if len(list_layer)>=1:
        init.normal_(net.hidden.weight)
        net.hidden.weight = torch.nn.Parameter(net.hidden.weight*0.01)
#         init.kaiming_normal_(net.batchnorm_1,mode='fan_in')
    if len(list_layer)>=2:
        init.normal_(net.hidden_2.weight)
        net.hidden_2.weight = torch.nn.Parameter(net.hidden_2.weight*0.01)
#         init.kaiming_normal_(net.batchnorm_2.weight,mode='fan_in')
    if len(list_layer)>=3:
        init.normal_(net.hidden_3.weight)
        net.hidden_3.weight = torch.nn.Parameter(net.hidden_3.weight*0.01)
#         init.kaiming_normal_(net.batchnorm_3.weight,mode='fan_in')
    if len(list_layer)>=4:
        init.normal_(net.hidden_4.weight)
        net.hidden_4.weight = torch.nn.Parameter(net.hidden_4.weight*0.01)
#         init.kaiming_normal_(net.batchnorm_4.weight,mode='fan_in')
#     init.kaiming_normal_(net.batchnorm.weight,mode='fan_in')
#     init.kaiming_normal_(net.w_allocate.weight, mode='fan_in')
#     net.w_allocate.weight.data.fill_(1)
    init.normal_(net.output.weight)
    net.output.weight = torch.nn.Parameter(net.output.weight*0.01)
    
    
def run_(net,optimizer,tensor_input,list_layer,gamma_A,state=False):
    
#     optimizer = torch.optim.Adam(net.parameters(), lr=0.002,weight_decay=0)  # 传入 net 的所有参数, 学习率
#     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300,600], gamma=0.3)
        
    list_loss = []
    list_grad = []
    for t in range(402):
        loss,F_is,weight_3200,x_3200,mid_x = net(tensor_input,print_state = state)
        state = False
    #     loss = loss_func(predicted_r,tensor_y.float32) 
        
        loss = loss+ gamma_A*penalty_adding(net,list_layer)
        
        list_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
#         print('input grad:',tensor_input.grad)
#         scheduler.step()

        if t % 50 == 0:
            print(t)
            print('loss:',loss,'penalty:',gamma_A*penalty_adding(net,list_layer))
            state = True
        
        if t>352:
            # 2022.11.16  new added:
            if len(list_layer) == 1:
                list_grad.append((net.output.weight.grad @ net.hidden.weight.grad).data.numpy())
            elif len(list_layer) == 2:
                list_grad.append((net.output.weight.grad@ net.hidden_2.weight.grad@ net.hidden.weight.grad).data.numpy())
            elif len(list_layer) == 3:
                list_grad.append((net.output.weight.grad@ net.hidden_3.weight.grad@ net.hidden_2.weight.grad@ net.hidden.weight.grad).data.numpy())
#             list_grad.append(net.hidden.weight.grad.mean(axis=0).data.numpy())
            
    return list_loss,F_is,weight_3200,x_3200,list_grad,mid_x


    
def set_seed(manualSeed=666):
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    
list_bond = [
        'rating','duration', 'VaR_5%','Amihud','1-month_mom',

        'ytm',  'size', 'age', 'time2maturity',  'turnover',  

        'VaR_10%',

        'std_Amihud', 'Roll', 'BPW', 'P_HL',
        'P_FHT',  'TC_IQR', 'Range_daily', 'trades', 

        'variance',
        'skewness', 'kurtosis', 

        'COSKEW', 'ISKEW', 
        'market_beta', 'market_residual_variance', 
        'term_beta', 'default_beta', 'term_default_residual_variance', 
        'drf_beta', 'crf_beta', 'lrf_beta',
        'liq_beta', 'vix_beta', 'unc_beta', 
        '6-month_mom', '12-month_mom', 'LTR_mom',
    'barQ','std_barQ_1mom','range_monthly',
]

list_equity = ['adm','bm_ia','herf','hire','me_ia','baspread','beta','ill','maxret','mom12m','mom1m',
 'mom36m',
 'mom60m',
 'mom6m',
 're',
 'rvar_capm',
 'rvar_ff3',
 'rvar_mean',
 'seas1a',
 'std_dolvol',
 'std_turn',
 'zerotrade',
 'me',
 'dy',
 'turn',
 'dolvol',
 'abr',
 'sue',
 'cinvest',
 'nincr',
 'pscore',
 'acc',
 'bm',
 'agr',
 'alm',
 'ato',
 'cash',
 'cashdebt',
 'cfp',
 'chcsho',
 'chpm',
 'chtx',
 'depr',
 'ep',
 'gma',
 'grltnoa',
 'lev',
 'lgr',
 'ni',
 'noa',
 'op',
 'pctacc',
 'pm',
 'rd_sale',
 'rdm',
 'rna',
 'roa',
 'roe',
 'rsup',
 'sgr',
 'sp']
list_option = ['ivslope', 'ivvol', 'ivrv', 'ivrv_ratio', 'atm_civpiv',
       'skewiv', 'ivd', 'dciv', 'dpiv', 'atm_dcivpiv', 'nopt', 'so', 'dso',
       'vol',  'pba', 'pcratio', 'toi', 
        'mfvu', 'mfvd', 'rns1m', 'rnk1m', 'ivarud30', 'rns3m',
       'rnk3m', 'rns6m', 'rnk6m', 'rns9m', 'rnk9m', 'rns12m', 'rnk12m']
# list_char = list_bond+list_equity