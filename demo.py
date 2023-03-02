import numpy as np
import pandas as pd
import statsmodels.api as sm
import imp
import os
import statsmodels.api as sm
from DL_functions_demo import *

from pandas.tseries.offsets import MonthEnd

port_type = 'ew'
bm_type = 'CAPM'
char_type = 'bond_equity_option'
i=1

# read in the risk-free rate

df_rf_all = pd.read_table('../data_in_all/one_month_bill_2020.txt', sep = '\s+', dtype = {'caldt': str})
df_rf_all = df_rf_all.rename(columns = {'caldt': 'trd_exctn_dt'}).set_index('trd_exctn_dt')
df_rf_all.index = pd.to_datetime(df_rf_all.index, format = '%Y%m%d') + MonthEnd(0)
df_rf_all = df_rf_all.loc['2004-07-31':]

# characteristics and return data
df_char_all = pd.read_feather(
    '../data_in_all/characteristics_impute_selected3200_with_Equity_0929_xret.feather').set_index('trd_exctn_dt')
## the below df_mkt_all is ew version

# EW method to calculate the market return
if port_type == 'ew':
    df_mkt_all = df_char_all.groupby('trd_exctn_dt').mean()['excess_ret'] 
elif port_type == 'vw':
    df_mkt_all = df_char_all.groupby('trd_exctn_dt').apply(cal_portfolio_vw_return)

df_mkt_all = np.array(df_mkt_all)
df_return_all = np.array(df_char_all['excess_ret']).reshape((198, -1))
df_xret_all = df_return_all 


if char_type == 'bond':
    list_char = list_bond
    df_char_all = df_char_all.loc[:, list_char]
elif char_type == 'bond_equity_option':
    list_char = list_bond + list_equity+list_option
    df_char_all = df_char_all.loc[:, list_char]
elif char_type == 'bond_equity':
    list_char = list_bond + list_equity
    df_char_all = df_char_all.loc[:, list_char]

df_char_all_2 = df_char_all.copy()
df_char_all = np.array(df_char_all).reshape((198, 3200, -1))

T_IS = 120
T_OOS = 78

import pickle

num_char = len(list_char)
list_layer = [num_char//2,num_char//4,num_char//8,num_char//16]

dict_para = {}

# the pre-validated parameters
if char_type == 'bond_equity_option':
    dict_para['CAPM'] = np.array([[5*1e-2,1e-7,1e-8],[1*1e-2,1e-5,1e-7],[5*1e-2,1e-8,1e-8]])

dict_bm = {}
dict_bm['CAPM'] = df_mkt_all.reshape(-1,1)


data_input = dict(bond_return = torch.from_numpy(df_xret_all[:120,:]).float(),
                  bond_return_oos = torch.from_numpy(df_xret_all[120:,:]).float(),
                  num_firm = df_char_all.shape[1],
                  num_feature = num_char
                 )

tensor_input = torch.from_numpy(df_char_all[:120,:,:]).float()
tensor_input_oos = torch.from_numpy(df_char_all[120:,:,:]).float()

df_benchmark = dict_bm[bm_type][:120,:]
        

    
#######

#######        
seed = 666
set_seed(seed)


for n_layer in range(i, i + 1):
    learning_rate = dict_para[bm_type][i-1][0]
    gamma_A = dict_para[bm_type][i-1][1]
    gamma_l2 = dict_para[bm_type][i-1][2]
    print('lr:',learning_rate,'gamma_A:',gamma_A)
    input_layer = list_layer[:n_layer].copy()
    data_input['layers'] = input_layer
    for n_deep_factors in range(1, 4):
        if n_deep_factors >= 2:
            temp = torch.cat([data_input['factor'],F_is],axis=1)
            data_input['factor'] = torch.from_numpy(temp.data.numpy()).float()
        else:
            data_input['factor'] =  torch.from_numpy(df_benchmark).float()
            

        net = Net_SR(data_input)

        initialization_(net,input_layer)
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,weight_decay=gamma_l2)

        list_loss,F_is,weight_3200,x_3200,list_grad,_ = run_(net,optimizer,tensor_input,input_layer,gamma_A)

        F_oos,weight_3200_oos,x_3200_oos,_ = net(tensor_input_oos,sample_period='oos')
        F_whole_np = torch.cat([F_is,F_oos]).data.numpy()
        x_all = torch.cat([x_3200,x_3200_oos]).data.numpy()
        w_all = torch.cat([weight_3200,weight_3200_oos]).data.numpy()
        print(w_all.shape)
        
        target_file = 'demo_results'.format(char_type,bm_type)
        if not os.path.exists(target_file):
            os.mkdir(target_file)


        np.savetxt(target_file+
            '/factor_{}_{}_gammaA{}_{}.txt'.format(n_layer,n_deep_factors,gamma_A, seed), F_whole_np)

        np.savetxt(target_file+
            '/loss_path_{}_{}_gammaA{}_{}.txt'.format(n_layer,n_deep_factors,gamma_A, seed), list_loss)

        np.savetxt(target_file+
            '/weights_{}_{}_gammaA{}_{}.csv'.format(n_layer,n_deep_factors,gamma_A, seed), w_all)

        np.savetxt(target_file+
            '/dchar{}_{}_gammaA{}_{}.txt'.format(n_layer,n_deep_factors,gamma_A, seed), x_all)


        torch.save(net, target_file+'/torchnn_{}_{}_gammaA{}_{}.pt'.format(n_layer,n_deep_factors,gamma_A, seed))

        print('save_success')