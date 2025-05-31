#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
base = pd.read_csv('D:/Users/tonyn/Desktop/da_sci_4th/datathon/DATA/Base.csv')
var1 = pd.read_csv('D:/Users/tonyn/Desktop/da_sci_4th/datathon/DATA/Variant I.csv')
var2 = pd.read_csv('D:/Users/tonyn/Desktop/da_sci_4th/datathon/DATA/Variant II.csv')
var3 = pd.read_csv('D:/Users/tonyn/Desktop/da_sci_4th/datathon/DATA/Variant III.csv')
var4 = pd.read_csv('D:/Users/tonyn/Desktop/da_sci_4th/datathon/DATA/Variant IV.csv')
var5 = pd.read_csv('D:/Users/tonyn/Desktop/da_sci_4th/datathon/DATA/Variant V.csv')
#%%
base_copy = base.copy()
var1_copy = var1.copy()
var2_copy = var2.copy()
var3_copy = var3.copy() 
var4_copy = var4.copy()
var5_copy = var5.copy()
# %%
### 전처리 ###
data = [base_copy, var1_copy, var2_copy, var3_copy, var4_copy, var5_copy]
drop_cols = ['days_since_request','payment_type', 'employment_status', 'prev_address_months_count', 'intended_balcon_amount', 'housing_status']
for df in data:
    df.drop(columns=drop_cols, inplace=True, errors='ignore')
    df['bank_months_count'].replace(-1, 0, inplace=True)
    df[:] = df[df['session_length_in_minutes'] != -1]
    df[:] = df[df['current_address_months_count'] != -1]
    # df['days_since_request'].replace(-1, 0, inplace=True)
