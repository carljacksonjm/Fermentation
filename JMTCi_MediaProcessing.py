#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns

# import matplotlib.pyplot as plt
# import os, tempfile
# import xlrd
# import matplotlib.dates as md
# import scipy
# import warnings
# warnings.filterwarnings('ignore')
# from PlottingFunctions import *

# figure_dir = "../../Pictures/FigureDump/"

jm_colors_dict = {
    'JM Blue':'#1e22aa',
    'JM Purple':'#e50075',
    'JM Cyan':'#00abe9','JM Green':'#9dd3cb',
    'JM Magenta':'#e3e3e3',
    'JM Light Grey':'#575756',
    'JM Dark Grey':'#6e358b'
}

jm_colors_list = list(jm_colors_dict.values())
cb_colors_list = sns.color_palette("muted")+sns.color_palette("muted")  # deep, muted, pastel, bright, dark, and colorblind
hls_colors_list = sns.color_palette("hls", 8) + sns.color_palette("hls", 8)
tab10_colors_list = sns.color_palette("tab10") + sns.color_palette("tab10")

marker_list = ["o","v","D","x","X","<","8","s","^","p","P","*","h","H","+","d","|","_"]

#%% functions
def CamelCase(col_bad):
    
    good_col = col_bad.title().replace(" ","_")
    
    return good_col.upper()


# In[2]:


#%% input variables
source_path = "C:/Users/JacksC01/OneDrive - Johnson Matthey/Documents/DigitalModelling/JMTCi4Fermentation/data/media.csv"

output_path = Path("C:/Users/JacksC01/OneDrive - Johnson Matthey/Documents/DigitalModelling/JMTCi4Fermentation/data")
output_fname = "trends_processed.csv"

#%% data import
df_m = pd.read_csv(source_path)

tm_names = df_m["Media Name"].loc[df_m["Media Type"] == "TE"].unique()


# In[6]:


#%% data processing
df_all = df_m.pivot(
    index=["Media Name"],
    columns=["Component"],
    values=["Target Conc (g/L)","Target Conc (%, v/v)"])

# multilevel col names to single
df_all.columns = [col[1] for col in df_all.columns] 
# display(df_all)


# In[7]:


# splitting df into base and tm
df_tm = df_all.loc[df_all.index.isin(tm_names)]
df_b = df_all.loc[~df_all.index.isin(tm_names)]

all_dfs = [df_tm, df_b]

for d in all_dfs:
    d.dropna(how = "all", axis = 1, inplace = True) 
    d.columns = [CamelCase(col) for col in d.columns]
    d.index = [CamelCase(i) for i in d.index]


# In[8]:


# column identifying tm name
# removing redundant cols
tm_cols = ['TE_SOLUTION_(TE_MOD2)', 'TE_SOLUTION_MOD1',]
df_b["TM_SOLUTION"] = np.nan # initialise column
df_b['TM_TARGET_CONC_(%,_V/V)'] = np.nan

tm_drop_cols = []
for tm_col in tm_cols:
    tm_drop_cols.append(tm_col)
    
    df_b["TM_SOLUTION"] = np.where(df_b[tm_col].notnull(), tm_col, df_b["TM_SOLUTION"])
    df_b['TM_TARGET_CONC_(%,_V/V)'] = np.where(df_b[tm_col].notnull(), df_b[tm_col], df_b['TM_TARGET_CONC_(%,_V/V)'])

df_b.drop(columns = tm_drop_cols, inplace = True)
# display(df_b)


# In[9]:


#%% scaling and joining of tm to base df
# scaling because tm components are in g/l of tm
# scaling to be g/l of base
df_tm.rename(columns = {"HYDROCHLORIC_ACID_>35%":'HYDROCHLORIC_ACID_>35%_(%,_V/V)'}, inplace = True)

tm_names = df_tm.index.unique()

df_tm_scaled = df_tm

for tm in tm_names:
    scale_val = (df_b["TM_TARGET_CONC_(%,_V/V)"].loc[df_b["TM_SOLUTION"] == tm].values[0])/100
    df_tm_scaled.loc[tm] = df_tm_scaled.loc[tm] * scale_val# df_tm.loc[tm].multiply(scale_val, axis = 1)


# In[10]:


#%% joining trace metal df with base media
df_all = pd.merge(df_b, df_tm_scaled,
                  how='outer',
                  left_on='TM_SOLUTION', right_index=True)

#%% exporting
df_all.to_csv(output_path / output_fname)


# In[ ]:




