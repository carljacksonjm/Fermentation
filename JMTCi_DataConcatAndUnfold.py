"""
base pump , contin / discrete 
some media better for growth than others?
optimal pcas for each stage
pc values for each stage
"""
#%% imports
from tkinter import N
from matplotlib.markers import MarkerStyle
from numpy.lib.npyio import save
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import math
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import warnings

#from JMTCi_BatchWiseCleverStuff import basic_scatter
warnings.filterwarnings('ignore')

jm_colors_dict = {
    'JM Blue':'#1e22aa',
    'JM Purple':'#e50075',
    'JM Cyan':'#00abe9','JM Green':'#9dd3cb',
    'JM Magenta':'#e3e3e3',
    'JM Light Grey':'#575756',
    'JM Dark Grey':'#6e358b'
}

jm_colors_list = list(jm_colors_dict.values())
# cb_colors_list = sns.color_palette("muted")+sns.color_palette("muted")  # deep, muted, pastel, bright, dark, and colorblind
# hls_colors_list = sns.color_palette("hls", 8) + sns.color_palette("hls", 8)
tab10_colors_list = sns.color_palette("tab10") + sns.color_palette("tab10")
marker_list = ["o","v","D","x","X","<","8","s","^","p","P","*","h","H","+","d","|","_"]
linestyle_list = ['-','--', '-.','-','--', '-.','-','--', '-.']

#%% functions
def clean_col(bad_col):

    def replace_all(text, dic):
        for i, j in dic.items():
            text = text.replace(i, j)
        return text

    replace_dict = {"]":"",
                    "_":" ",
                    ",":" ",
                    "-":" ",
                    ".":" "
                    }

    #.replace(" ","_").replace("]","").replace(" ","")
    good_col = replace_all(bad_col, replace_dict)
    good_col = good_col.upper().replace(" ","_")

    return good_col

def calibrate_pca(X_vals, n_components):

    pca = PCA(n_components = n_components)
    pca_scores = pca.fit_transform(X_vals) # PCA scores
    pca_loadings = pca.components_.T # eigenvectors  aka loadings
    eigen_values = pca.explained_variance_ # eigenvalues
    print(f"X values: {X_vals.shape}")
    print(f"Eigenvectors (loadings): {pca_loadings.shape}\nEigenvals: {eigen_values.shape}\nScores: {pca_scores.shape}")
    
    return pca_loadings, pca_scores, eigen_values

def check_day_month_order(date_series, identifier, list_month_first):

    date_series_clean = date_series.apply(lambda x: str(x).replace("-","/"))

    if identifier in list_month_first:
        print(f"Month first {identifier}")

        return pd.to_datetime(date_series_clean, format='%Y/%d/%m %H:%M:%S')
    else:
        #print("day first apparently: ", date_series_clean[:3])
        return pd.to_datetime(date_series_clean, dayfirst= True) # , format='%d/%m/%Y %H:%M:%S'

def strings_to_floats(string_series):
    """converts to string
    replaces ' and ,
    converts to float    
    """

    string_series = str(string_series)
    clean_string = string_series.replace("'","").replace(",","")
    return float(clean_string)


def clean_nasty_floats(nasty_series):
    good_series = nasty_series.replace("\t","", regex = True)
    good_series = good_series.replace(" ","", regex = True)

    return pd.to_numeric(good_series)

def day_counter(row_val):
    global day_count

    if row_val < 0:
        day_count += 1
    
    return day_count

def remove_date(row_str):
    if " " in row_str:
        row_str = str(row_str).split(" ")[-1]
    return str(row_str)

def sns_barplot(dataFrame, x_label, y_label, orient_var = "v",
title_var = None, save_path = None):
    
    fig, ax = plt.subplots(figsize = (6,6))
    
    sns.barplot(x=x_label, y=y_label, data=dataFrame, ax=ax, palette = tab10_colors_list, edgecolor = "black", orient = orient_var)

    # plt.grid(axis = "y")
        
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    
    if title_var:
        fig.suptitle(title_var)
        
    plt.tight_layout()
    
    if save_path:
        image_name = save_path / (title_var.replace(" ", "")+".png")
        plt.savefig(image_name, facecolor='w')
     
    plt.show()

def contribution_chart(data_variables, loadings, x_label, xlim = None, title_var = None,
save_path = None, save_fname = None):
    
    fig, ax = plt.subplots(figsize = (12,6))
    
    sns.barplot(x = loadings, y = data_variables, edgecolor = "black", orient = "h", color = tab10_colors_list[0], ax = ax)
    ax.set_xlabel(x_label)
    ax.set_ylabel("")

    if xlim:
        plt.xlim(xlim)
        # dxlim = (max(xlim) - min(xlim))/8
        # plt.xticks(np.arange(min(xlim), max(xlim)+dxlim, dxlim))

    plt.title(title_var)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_color('none')
    plt.grid(which = "both", axis = "x")
    plt.tight_layout()
    
    if save_path:
        print("saving...")
        if save_fname:
            image_name = save_path / (save_fname + ".png")
        else:
            image_name = save_path / (title_var.replace(" ", "")+".png")
        
        plt.savefig(image_name, facecolor='w')
     
    plt.show()

def pca_var_explained(X_array, number_components, x_lim = [0,10],title_var = None, save_path = None):

    cov_mat = np.cov(X_array.T) # covariance matrix
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat) # decom
    tot = sum(eigen_vals)
    var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    print(f"{number_components} components exlain {cum_var_exp[number_components-1]*100: .1f} % of total variation")

    # plotting
    image_name = title_var.replace(" ", "") + ".png"
    fig, ax = plt.subplots(figsize = (6,6))
    x_vals = range(1, len(cum_var_exp)+1)

    ax.bar(x_vals, var_exp,
            alpha=1, align='center',
            label='Individual', color = 'grey')

    ax.step(x_vals, cum_var_exp, where = "mid", label='Cumulative', color = "black")

    plt.xlim(x_lim)
    plt.ylim([0,1])
    plt.ylabel('Relative Variance Ratio')
    plt.xlabel('Principal Components')
    plt.legend(loc='best')
    plt.grid(which = "both", axis = "y")

    if title_var:
        plt.title(title_var)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path / image_name, facecolor='w', dpi = 100)

    plt.show()

def time_var_loading_plot(x_vals, y_vals, c_vals,
x_label, c_label, title_var = None, save_fname = None, save_path = None):
 
    fig, ax = plt.subplots(figsize = (10,8))
    
    color_scale_lim = c_vals.abs().max()

    im = ax.scatter(x_vals,
    y_vals, c = c_vals,
    cmap = "coolwarm_r", marker = 's', s = 200, alpha = 0.5,
    edgecolor = None, vmin = -1*color_scale_lim, vmax = color_scale_lim)

    fig.colorbar(im, ax=ax, label = c_label)
    ax.set_ylabel("")
    ax.set_xlabel(x_label)
    ax.grid(which = "both", axis = "x")
      
    if title_var:
        plt.title(title_var)
        
    plt.tight_layout()
    
    if save_path:
        if save_fname == None:
            save_fname = title_var.replace(" ","")
        else:
            pass

        image_name = save_path / (save_fname.replace(" ", "")+".png")
        plt.savefig(image_name, facecolor='w')
    
    plt.show()

def parity_plot(y_val_train, y_val_train_pred,
x_label, y_label, title_var, y_val_test = None, y_val_test_pred = None, save_path = None,
alpha_var = 0.7, marker_s_var = 70, zero_axis_lim = True):
    # plt.style.use('default')

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(y_val_train,
    y_val_train_pred, color = tab10_colors_list[0],
    alpha=alpha_var, s = marker_s_var, label = f"Train")

    if y_val_test is not None:
        all_ys = list(y_val_train) + list(y_val_train_pred)+\
        list(y_val_test_pred) + list(y_val_test)

        ax.scatter(y_val_test,
        y_val_test_pred, color = tab10_colors_list[1],
        alpha=alpha_var, s = marker_s_var, label = f"Test")

        plt.legend(loc = "upper left")
    
    else:
        all_ys = list(y_val_train) + list(y_val_train_pred)
    
    if zero_axis_lim == False:
        y_lims = [np.round(max(all_ys)*-1.1), np.round(max(all_ys)*1.1, 2)] # np.round(min(all_ys)*0.9, 2)
    else:
        y_lims = [0, np.round(max(all_ys)*1.1, 2)]

    ax.plot(y_lims, y_lims, linestyle = "--", color = "grey")    

    ax.set(xlabel=x_label, ylabel= y_label, title= title_var)
    
    ax.set_ylim(y_lims)
    ax.set_xlim(y_lims)

    plt.grid(which = "both", axis = "both")
    plt.tight_layout()

    if save_path:
        image_name = save_path / (title_var.replace(" ", "")+".png")
        plt.savefig(image_name, facecolor='w')

    plt.show()

def vol(d, h):
    vol = round((math.pi*d**2)/4 * h,5)*1000 # l
    asp_ratio = round(h/d,3)
    print(vol, asp_ratio)
    return vol, asp_ratio

def tip_speed(d, N):

    return math.pi * d * N

def P_per_V(N, d, V, Np, n_imps = 1, rho = 1e3):
    """
    N: Agiator speed Revs/s
    d: Impeller diameter (m)
    V: Liquid volume (m3)
    rho: Density (kg/m3)
    Np: Power number
    https://www.eppendorf.com/product-media/doc/en/633183/Fermentors-Bioreactors_Publication_BioBLU-c_Cell-Culture-Scale-Up-BioBLU-Single-Vessels.pdf
    Np = 1.27 for PBT, 5 for rushton
    n_imp: Number of impellers with sufficient spacing to be counted individually
    """
    P_per_V = n_imps * (Np * rho * (N)**3 * d**5 ) / V # W / m3

    return P_per_V

def reynolds_mixing(d, N, rho = 1e3, mu = 8.9e-4):
    """Reynolds number in mixing
    d: Impeller diameter (m)
    N: Agiator speed Revs/s
    rho: Density (kg/m3)
    mu: Dynamic viscosity (Pa s)
    """
    return d*N*rho/mu

def reynolds_chech(Re):
    
    if Re > 4000:
        return "Turbulent"
    elif Re < 2100:
        return "Laminar"
    else:
        return "Transitional"

#%% input variables
#%% input variable
source_path = Path("C:/Users/JacksC01/OneDrive - Johnson Matthey/Documents/DigitalModelling/JMTCi4Fermentation/data/trends")
summary_path = Path("C:/Users/JacksC01/OneDrive - Johnson Matthey/Documents/DigitalModelling/JMTCi4Fermentation/data")

output_path = Path("C:/Users/JacksC01/OneDrive - Johnson Matthey/Documents/DigitalModelling/JMTCi4Fermentation/figures")
processed_path = summary_path

summary_fname = "Production summary for Carl.xlsx"

output_fname1 = "trends_processed.csv"
output_fname2 = "obswise_fermentation.csv"
output_fname3 = "batchwise_fermentation.csv"

export_cols = True
plot_graphs = True

impeller_d_dict = { # vol (l) : diam (m)
    1: 0.038, 42: 0.089,
    0.85: 0.038, 27.5: 0.089
    }
vol_scale_dict = { # vol (l) : vol (m)
    1: 0.85, 42: 27.5
    }

n_impeller_dict = { # vol (l) : vol (m)
    1: 2, 42: 3
    }

#n_impellers = 3

# 750 l
d = 300 # mm
s = 550
print(s/d)

dt = 0.25
n_exp = 0.33#0.47
A_exp = 11.249370177870125 # A*ug**beta*dCO2
t_rolling = 2 #hours
q0 = 0.08


#%% import of production summary and clean
df_ps = pd.read_excel(summary_path/summary_fname, sheet_name = "Sheet1")
df_ps.dropna(how = "all", axis = 1, inplace = True)

df_ps["USP No"] = df_ps["USP No"].apply(
    lambda x: str(x.replace("USP",""))
    )
df_ps.rename(columns = {"USP No":"USP"},inplace = True)
df_ps.columns = [col.replace("x̅","x_bar").replace("[","(").replace("]",")") for col in df_ps.columns]
ps_cols = df_ps.columns.to_list()
df_ps.set_index("USP", drop = True, inplace = True)

df_ps["Scale"] = df_ps["Scale"].apply(lambda x: int(x.split("L")[0]))
df_ps["Impeller Diameter (m)"] = df_ps["Scale"].map(impeller_d_dict)
df_ps["Volume (l)"] = df_ps["Scale"].map(vol_scale_dict)
df_ps["Number of Impellers"] = df_ps["Scale"].map(n_impeller_dict)

df_ps["Feed Media"] = df_ps["Feed Media and sequence"].apply(lambda x: x.split("_")[0])
df_ps["Feed Sequence"] = df_ps["Feed Media and sequence"].apply(lambda x: x.split("_")[1])

seq_red_dict = {
    "001":0,
    "010":20,
    "009":25,
    "008":50,
    "011":20
}

seq_type_dict = {
    "001":"Steps",
    "010":"Steps",
    "009":"Steps",
    "008":"Steps",
    "011":"Ramp"
}

df_ps["Feed Reduction at Induction (%)"] = df_ps["Feed Sequence"].map(seq_red_dict)
df_ps["Feed Rate Fraction at Induction"] = (100 - df_ps["Feed Reduction at Induction (%)"])/100

df_ps["Stepped Feed Sequence"] = df_ps["Feed Sequence"].map(seq_type_dict)
df_ps["Biomass Density (g/l)"] = df_ps['Total Biomass (g)'] / df_ps['Volume (l)']

#
ps_x_vars = ["Feed Media", "Glycerol ID","Production Media","Scale",
"Induction Temp (°C)","Induction pH", "Antibiotic",
"Feed Rate Fraction at Induction", "Stepped Feed Sequence"]

features = ['Volumetric Activity (U/mL)', 'Harvest x_bar WCW (g/L)',
"Biomass Density (g/l)"]

df_ps_clean = df_ps[ps_x_vars + features]
dummy_cols = df_ps_clean.select_dtypes(include = ["object"]).columns.to_list()
df_ps_clean = pd.get_dummies(df_ps_clean, prefix = dummy_cols, prefix_sep = "_")

dupe_cols =  ['Glycerol ID_WGS-0020','Antibiotic_None', "Stepped Feed Sequence_Ramp"]
df_ps_clean.drop(columns = dupe_cols, inplace = True)

X_cols_ps = list(set(df_ps_clean.columns) - set(features))
print(df_ps_clean.columns.to_list())
# total acid and base
df_nans = df_ps.groupby(['Scale','Volume (l)',"DOI (dd/mm/yr)",'USP'])[['25% ammonia used (ml)','2M H3PO4 used (ml)','Impeller Diameter (m)',"Number of Impellers"]].mean()

#%%
final_acid = 0.5 # (ml)
final_base = 85 # (ml)
print(f"""Filling missing final base values with {final_base} ml
Filling missing final acid values with {final_acid} ml""")
print(df_nans)
df_nans["25% ammonia used (ml)"].fillna(value = final_base, inplace = True)
df_nans["2M H3PO4 used (ml)"].fillna(value = final_acid, inplace = True)

df_nans.reset_index(drop = False, inplace = True)
df_nans.set_index("USP", drop = True, inplace = True)

#%% import of trend data and clean
dict_of_dfs = {}
month_first_list = ["0785","0786","0790"]

csv_cols, xlsx_cols = set(), set()
csv_usps = []
csv_rename_cols = {
"Date Time UTC":"TIMESTAMP",
'pH, -':'pH',
'Temp, °C':'Temp',
'Temp, Â°C':'Temp',
'Stirrer, 1/min':'Stirrer',
'pO2, %':'pO2',
'Antifoam, -':'Anti_Foam',
'GasMix, %O2':'Gas_Mix',
'Feed.Total volume, ml':"Feed_Pump",
'Acid Pump.Total volume, ml':"Acid_Pump",
'Base Pump.Total volume, ml':"Base_Pump"
}
# FEED_RATE_
keep_cols = ['Temp', 'Stirrer', 'pH', 'pO2',
       'Anti_Foam', 'Feed_Rate', 'Gas_Mix', 'Acid_Pump', 'Base_Pump',
       'AF_Pump', 'Feed_Pump', 'USP', 'BATCH_TIME_H',
       'Base_Total', 'Acid_Total']

file_type_dict = {}

# iterates through all files in directory
for filename in os.listdir(source_path):
    
    if filename.lower().endswith(".xls"):
        file_type = ".xls"
        file_df = pd.read_excel(source_path/filename, sheet_name = "Iris report table 1")

    elif filename.lower().endswith(".csv"):
        file_type = ".csv"
        file_df = pd.read_csv(source_path/filename, delimiter= ";", skiprows=1)
        file_df.rename(columns = csv_rename_cols, inplace = True)
        file_df["LoggingTime"] = pd.to_datetime(file_df['Batch Time (since inoc.), sec'],
            unit='s').dt.strftime("%H:%M:%S")

    elif filename.lower().endswith(".xlsx"):
        file_type = ".xlsx"
        file_df = pd.read_excel(source_path/filename, sheet_name = "Iris report table 1")
        file_df.rename(columns = csv_rename_cols, inplace = True)

        file_df["Feed_Pump"] = clean_nasty_floats(file_df["Feed_Pump"])
        file_df["Feed_Pump"] = file_df["Feed_Pump"].cumsum()

    else:
        print(f"Unknown file {filename} ... breaking")
        break
    
    filename_short = filename.split(file_type)[0]
    usp = str(filename_short.replace("USP",""))

    file_df["TIMESTAMP"] = file_df["LoggingTime"].astype(str).apply(lambda x: remove_date(x)) 
    file_df["BATCH_TIME_M"] = pd.to_timedelta(file_df["TIMESTAMP"]).astype('timedelta64[m]')
    file_df["BATCH_TIME_H"] = file_df["BATCH_TIME_M"].apply(lambda x: round((x/60), 2))
    file_df["BATCH_TIME_DIFF"] = file_df["BATCH_TIME_H"].diff()

    day_count = 0
    file_df["DAY_COUNT"] = file_df["BATCH_TIME_DIFF"].apply(lambda x: day_counter(x))
    file_df["BATCH_TIME_H"] = file_df["BATCH_TIME_H"] + 24. * file_df["DAY_COUNT"]

    specific_keep_cols = list(set(keep_cols).intersection(set(file_df.columns)))
    file_df = file_df[specific_keep_cols]
    # convert to float cols
    float_cols = set(specific_keep_cols) - {"USP", "BATCH_TIME_H"}
    for col in float_cols:

        file_df[col] = clean_nasty_floats(file_df[col])

    file_df.dropna(subset = ['pH'], axis = 0, inplace = True)
    file_df = file_df.fillna(0)

    # resample
    file_df["DATETIME"] = file_df["BATCH_TIME_H"].apply(
        lambda x: pd.to_datetime("25/12/1900 00:00:00", dayfirst = True) + pd.DateOffset(hours=x)
        )
    file_df.set_index("DATETIME", inplace = True, drop = True)
    file_df = file_df.resample("15Min").mean()
    file_df.reset_index(drop = False, inplace = True)
    file_df["BATCH_TIME_H"] = (file_df["DATETIME"] - file_df["DATETIME"].min()).astype('timedelta64[m]')
    file_df["BATCH_TIME_H"]  = file_df["BATCH_TIME_H"] .apply(lambda x: round((x/60), 2))

    file_df["USP"] = usp
    file_type_dict[usp] = file_type
    # file_df["FILE_TYPE"] = file_type

    print(filename,"\n",file_df["BATCH_TIME_H"].max() - file_df["BATCH_TIME_H"].min())

    print(f"{file_df.shape}\n---")
    dict_of_dfs[filename_short] = file_df

print("Done.")

#%% data joining
raw_df = pd.concat(dict_of_dfs, ignore_index = True, join = "outer", sort=False)
raw_df.dropna(how = "all", axis = 1, inplace = True)
raw_df.dropna(how = "all", axis = 0, inplace = True)
print(f"Trend df shape {raw_df.shape}\n---")

# feature engineering
df_trends = raw_df.copy()#[all_cols]
df_trends.drop(columns = ["DATETIME"], inplace = True)
df_trends.columns = [clean_col(cols) for cols in df_trends.columns]
df_trends["FILE_TYPE"] = df_trends["USP"].map(file_type_dict)
df_trends.sort_values(by = ["USP", "BATCH_TIME_H"], ascending= True, inplace= True)
df_trends = df_trends.loc[df_trends["BATCH_TIME_H"] <= 45]

# convert no. pumps to ml of acid and base
df_trends_usp = df_trends.groupby(["USP"])["ACID_PUMP", "BASE_PUMP"].max()
df_trends_usp = pd.merge(left = df_trends_usp, right = df_nans, how = "left",
left_index = True, right_index = True)
df_nans

df_trends_usp["ACID_PUMP_RATE"] = df_trends_usp["2M H3PO4 used (ml)"] / df_trends_usp["ACID_PUMP"]
df_trends_usp["BASE_PUMP_RATE"] = df_trends_usp["25% ammonia used (ml)"] / df_trends_usp["BASE_PUMP"]
df_trends_usp.replace(np.inf, 0, inplace = True)

df_trends = pd.merge(left = df_trends, 
    right = df_trends_usp[["ACID_PUMP_RATE", "BASE_PUMP_RATE", "Scale", 'Volume (l)',"Impeller Diameter (m)","Number of Impellers"]],
    how = "left",left_on = "USP",right_index = True)
df_trends["ACID_TOTAL_CALC"] = (df_trends["ACID_PUMP"] * df_trends["ACID_PUMP_RATE"])
df_trends["BASE_TOTAL_CALC"] = (df_trends["BASE_PUMP"] * df_trends["BASE_PUMP_RATE"])
df_trends["ACID_TOTAL"] = df_trends["ACID_TOTAL_CALC"] / df_trends["Volume (l)"]
df_trends["BASE_TOTAL"] = df_trends["BASE_TOTAL_CALC"] / df_trends["Volume (l)"]
df_trends['ANTI_FOAM'] = df_trends.groupby(["USP"])['ANTI_FOAM'].cumsum()
df_trends[["ACID_TOTAL","BASE_TOTAL"]] = df_trends[["ACID_TOTAL","BASE_TOTAL"]].fillna(value = 0) 
df_trends["FEED_FRACTION"] = df_trends.groupby(["USP"])["FEED_PUMP"].apply(lambda x: x / x.max())
df_trends["Tip Speed (m/s)"] = tip_speed(df_trends["Impeller Diameter (m)"], df_trends['STIRRER']/60)
df_trends["Power (W/m3)"] = P_per_V(df_trends['STIRRER']/60, df_trends["Impeller Diameter (m)"], df_trends["Volume (l)"]/1000, 5, df_trends["Number of Impellers"])

#
from scipy import integrate
df_trends.sort_values(by = ["USP","BATCH_TIME_H"], inplace = True, ascending = True)

df_trends["OUR"] = A_exp * df_trends["Power (W/m3)"] ** n_exp 
df_trends["OU"] = df_trends["OUR"] * dt # (mmol) ?
df_trends["OU Total"] = df_trends.groupby(["USP"])["OU"].cumsum()

df_trends["dOUR"] = df_trends.groupby(["USP"])["OUR"].diff(periods = 1)
df_trends["dOUR"].fillna(0, inplace = True)
df_trends["dOUR/dt"] = df_trends["dOUR"]/dt

# binary series of if antifoam was used
antifoam_s = df_trends.groupby("USP")["ANTI_FOAM"].apply(lambda x: 1 if x.max() > 0 else 0)
antifoam_s.rename("Antifoam", inplace = True)
df_ps_clean = pd.merge(left = df_ps_clean, right = antifoam_s, left_index = True, right_index = True)

#
av_window = int(t_rolling / dt)
df_trends.sort_values(by = ["USP","BATCH_TIME_H"], ascending = True, inplace = True)

diff_cols = ["ACID_TOTAL", "BASE_TOTAL"]
for col in diff_cols:
    new_col = col + "_DIFF"
    df_trends[new_col] = df_trends.groupby(["USP"])[col].diff(periods = 1)
    df_trends[new_col] = df_trends[new_col].fillna(0)
    print(new_col)

moving_av_cols = ["ACID_TOTAL_DIFF","BASE_TOTAL_DIFF","dOUR/dt"]
for col in moving_av_cols:
    new_col = col + "_MA"
    grouped_moving_av = df_trends.groupby('USP')[col].rolling(av_window, min_periods=1).mean()
    df_trends[new_col] = grouped_moving_av.reset_index(level=0, drop=True)
    print(new_col)

scale_cols = ['ACID_TOTAL', "ACID_TOTAL_DIFF_MA", 'BASE_TOTAL', "BASE_TOTAL_DIFF_MA","dOUR/dt_MA","OU Total"]
df_trends[scale_cols] = MinMaxScaler().fit_transform(df_trends[scale_cols].to_numpy())

#%%
df_trends.drop(columns = ["FEED_PUMP","OU","dOUR/dt",
    "ACID_PUMP","ACID_TOTAL_CALC", "ACID_PUMP_RATE",
    "BASE_PUMP","BASE_TOTAL_CALC", "BASE_PUMP_RATE",
    'AF_PUMP',"FILE_TYPE",'BASE_PUMP','FEED_PUMP','GAS_MIX',
    "ACID_TOTAL_DIFF","BASE_TOTAL_DIFF","ANTI_FOAM","Number of Impellers"], inplace = True)

#"Scale","Impeller Diameter (m)",
# rename
rename_dict = {
    'USP':"USP",
    'PH':"pH",
    'PO2':"DO2 (%)",
    'GAS_MIX':"Feed O2 (%)",
    'STIRRER':"Stirrer (RPM)",
    'BATCH_TIME_H': "Time (h)",
    'TEMP':"Temperature (°C)",
    'ACID_TOTAL':"Acid Total",
    "ACID_TOTAL_DIFF_MA":"Acid (2h MA)",
    'BASE_TOTAL':"Base Total",
    "BASE_TOTAL_DIFF_MA":"Base (2h MA)",
    'FEED_FRACTION':"Feed Total",
    "dOUR/dt_MA":"dOUR/dt (2h MA)",
    "OU Total":"OU Total",
    'Scale': "Scale (l)",
    'Induction Temp (°C)':'Induction Temperature (°C)',
    'Induction Ph':'Induction pH',
    'Feed Seq Red At Induct (%)':"Induction Feed Seq (%)",
    'Feed Media F04':'F04 Feed Media',
    'Feed Media_F04':'F04 Feed Media',
    'Glycerol Id Mgs-0467': "MGS-0467 Glycerol ID",
    'Glycerol Id Wgs-0020': "WGS-0020 Glycerol ID",
    'Glycerol ID_MGS-0467':"MGS-0467 Glycerol ID",
    'Production Media Gmsmk': "GMSMK Prod Media",
    'Production Media Gmsmk Mod1': "GMSMK Mod1 Prod Media",
    'Production Media Gmsmk Mod2': "GMSMK Mod2 Prod Media",
    'Production Media_gMSMK mod2': "GMSMK Mod2 Prod Media",
    'Production Media_gMSMK mod1': "GMSMK Mod1 Prod Media",
    'Production Media_gMSMK': "GMSMK Prod Media",
    'Antibiotic_Kan': "Kan Antibiotic",
    'Antibiotic None': "No Antibiotic",
    'Feed Seq Step Ramp':"Ramp Feed Seq",
    'Feed Seq Step Steps': "Step Feed Seq",
    'Stepped Feed Sequence_Steps':"Stepped Feed Sequence",
    'Biomass Density (G/L)':'Biomass Density (g/l)',
    "Protein Concentration (G/Ml)":"Protein Concentration (g/ml)",
    'Volumetric Activity (U/mL)':'Volumetric Activity (U/ml)',
    'Harvest x_bar WCW (g/L)':'Harvest WCW (g/l)',
    'Biomass Density (g/l)':'Biomass Density (g/l)',
    'Protein Activity (U/mg)':'Protein Activity (U/mg)'
    }

df_trends.rename(columns = rename_dict, inplace = True)
df_trends = df_trends.fillna(0)#.drop(columns = bad_cols)
#df_all = df_trends.copy()

df_ps_clean.rename(columns = rename_dict, inplace = True)
df_ps.rename(columns = rename_dict, inplace = True) 

features = ['Volumetric Activity (U/ml)','Harvest WCW (g/l)',
'Biomass Density (g/l)']
process_params = ['Temperature (°C)',
 'pH',
 'DO2 (%)',
 'Acid Total',
 'Base Total',
 'Feed Total',
 'Tip Speed (m/s)',
 'Power (W/m3)',
 'Acid (2h MA)',
 'Base (2h MA)',
 "OU Total",
 "dOUR/dt (2h MA)"
]

time_invar_params = list(set(df_ps_clean.columns) - set(features + process_params))# + ["Scale"]

#%% observation-wise unfolding
batch_t = [4, 9, 12, 22, 27, 35, 45]
df_trends_clean = pd.merge(left = df_trends[["USP","Time (h)"]+process_params],
    right = df_ps_clean[time_invar_params + features],
how = "left",
left_on = "USP", right_index = True)

print(f"File observation-wise unfolding\nShape {df_trends_clean.shape}\n---")
df_trends_clean.to_csv(processed_path / output_fname2)

#%% batch-wise unfolding data
# df_trends_clean = df_trends_clean.loc[df_trends_clean["Time (h)"].isin(batch_t)]
dict_dfs_unfolded = {}
df_means = df_trends_clean.loc[df_trends_clean["Time (h)"].isin(batch_t)]
df_means = df_means.groupby(["USP","Time (h)"])[process_params].mean()

for param in process_params:
    
    df_param = df_means.unstack()
    df_param = df_param[param]
    df_param = df_param.add_prefix(f'{param}, t=')

    n_nulls = df_param.isna().sum().sum()
    if n_nulls > 0:
        print(param, " - Nulls ",n_nulls)

    dict_dfs_unfolded[param] = df_param

df1 = pd.concat(dict_dfs_unfolded, axis=1, ignore_index=False)
df1.columns = df1.columns.droplevel(0)
# df1.dropna(how = "any", axis = 1, inplace = True)
df1.dropna(how = "any", axis = 0, inplace = True)
print("Unfolded dataframe shape: ", df1.shape)

#%% merging datasets
df1 = pd.merge(left = df1, right = df_ps_clean,
how = "left",
left_index = True, right_index = True)

X_cols = list(set(df1.columns.to_list()) - set(features))

print("Trend data joined with Production Summary data.")

# check for nulls
print("Columns with nulls", df1.columns[df1.isna().any()].tolist())
# for col in df1.columns:
#     na_count = df1[col].isna().sum()

#     if na_count > 0:
#         print(col, " ", na_count)

print(f"File batch-wise unfolding data\nShape {df1.shape}\n---")
df1.to_csv(processed_path / output_fname3)

print("Done.")

#%% volumetric activity specific power trend
from scipy.stats import linregress
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

def sns_lineplot(dataFrame, x_label, y_label, color_label,
style_label = None, palette_var = 'viridis_r', legend_var = False, save_fname = None, save_path = None):
    
    fig, ax = plt.subplots(figsize = (11,5))
    
    sns.lineplot(x=x_label, y=y_label, hue = color_label,
        style = style_label, data=dataFrame, ax=ax,
        legend = True, palette = palette_var)

    ax.grid(axis = "both")

    if legend_var == True:
            # plt.legend(title = color_label,
            # loc="upper center", frameon = False, ncol = 6, 
            # bbox_to_anchor=(0.5, 1.3), borderaxespad=0.)
            plt.legend(title = color_label,
            loc="upper left", frameon = False, ncol = 1, 
            bbox_to_anchor=(1., .9), borderaxespad=0.)
    else:
        plt.legend([],[], frameon=False)
        
    ax.set(ylabel = y_label, xlabel = x_label)        
    plt.tight_layout()
    
    if save_path and save_fname:
        image_name = save_path / (save_fname.replace(" ", "")+".png")
        plt.savefig(image_name, facecolor='w')
    plt.show()

def scatter_2D(x_vals, y_vals, x_label, y_label, xlim = None, ylim = None, fig_size = (8,6), x_scale = "linear", c_val = None, c_label = None,
save_fname = None, save_path = None):
    
    fig, ax = plt.subplots(figsize = fig_size)

    if c_label:
        
        edgecolor_var = "black"
        im = ax.scatter(x_vals, y_vals, c = c_val,
            cmap = "viridis", marker = 'o', s = 100, alpha = 0.7,linewidth = 1.5,
            edgecolor = edgecolor_var, label = c_label)
            
        fig.colorbar(im, ax=ax, label = f"{c_label}")
    else:

        edgecolor_var = "black"#None
        im = ax.scatter(x_vals, y_vals, marker = 'o', s = 100,
            facecolor = "none", linewidth = 1.5,
            edgecolor = edgecolor_var)
    
    ax.set(ylabel = y_label, xlabel = x_label, xscale = x_scale, xlim = xlim, ylim = ylim)
    # plt.xlim(xlim) 
    plt.tight_layout()

    # plt.grid(which = "both", axis = "both")
    
    if save_path and save_fname:
        image_name = save_path / (save_fname.replace(" ","") + ".png")
        
        plt.savefig(image_name, facecolor='w')
     
    plt.show()

def lin_reg(x_vals, y_vals):
    m, c, r_value, p_value, std_err = linregress(x_vals, y_vals)
    x_func = np.linspace(x_vals.min(), x_vals.max(), 50)
    y_func = m * x_func + c
    y_pred = m * x_vals + c
    r2_pred = r2_score(y_vals, y_pred)
    rmse_pred = abs(mean_squared_error(y_vals, y_pred))**0.5

    return m, c, std_err, r2_pred, rmse_pred, x_func, y_func

df_test = df_trends_clean.loc[df_trends_clean["Feed Total"]>0.025]
df_test.reset_index(drop = True, inplace = True)
df_test_2 = df_test.iloc[df_test.groupby('USP')["Feed Total"].agg(pd.Series.idxmin)]
#df_test_2 = df_test.iloc[df_test.groupby('USP')["Power (W/m3)"].agg(pd.Series.idxmax)]


m, c, std_err, r2_pred, rmse_pred, x_func, y_func = lin_reg(df_test_2["Time (h)"], df_test_2["Volumetric Activity (U/ml)"])

df_test_2 = df_test_2.join(df1[["Power (W/m3), t=9.0", "Power (W/m3), t=12.0"]], on = ["USP"], how = "left")

scatter_2D(df_test_2["Time (h)"], df_test_2["Volumetric Activity (U/ml)"], "Feed Start Time (h)", "Volumetric Activity (U/ml)",
save_fname = "VolumetricActivity_FeedStartTime", save_path = output_path)
scatter_2D(df_test_2["Power (W/m3), t=9.0"], df_test_2["Volumetric Activity (U/ml)"], "Power (W/m3), t=9.0", "Volumetric Activity (U/ml)",
save_fname = "VolumetricActivity_P9", save_path = output_path)

scatter_2D(df_test_2["Time (h)"], df_test_2["Power (W/m3), t=9.0"], "Feed Start Time (h)","Power (W/m3), t=9.0", 
c_val = df_test_2["Volumetric Activity (U/ml)"], c_label = "Volumetric Activity (U/ml)")

#save_fname = "VolumetricActivity_P9", save_path = output_path)

#c_val = df_test_2["Antifoam"], c_label = "Scale (l)")

sns_lineplot(df_test, "Time (h)", "Power (W/m3)", "Volumetric Activity (U/ml)",
style_label = None, legend_var = True, save_fname = "VolumetricActivity_PowerTrend",
save_path = output_path)

# sns_lineplot(df_test, "Time (h)", "Power (W/m3)", "Volumetric Activity (U/ml)",
# style_label = None, legend_var = True, save_fname = None,
# save_path = output_path)


#%% feed rate fraction
fig, axs = plt.subplots(1,3, figsize = (18,6))

for i, f in enumerate(features):

    sns.pointplot(data = df_ps, ax = axs[i],
    x = 'Feed Rate Fraction at Induction', y = f,
    hue = 'Production Media', palette = tab10_colors_list,
    dodge = True, markers = ["o","^","x"], hue_order = ['gMSMK','gMSMK mod1','gMSMK mod2'],
    errwidth = 1.5, capsize = 0.1)

    plt.tight_layout()
    if i == 0:
        axs[i].legend(title = "Production Media", frameon = True, framealpha = 1,
        loc = "best")
    else:
        axs[i].legend([], [])

plt.savefig(output_path / f"ProdMedia.png", facecolor='w')
plt.show()

#%% antibiotic
fig, axs = plt.subplots(1,3, figsize = (18,6))

for i, f in enumerate(features):

    sns.barplot(data = df_ps.loc[df_ps["Feed Rate Fraction at Induction"]>=0.8], ax = axs[i],
    x = "Antibiotic", y = f,
    hue = "Production Media", hue_order = ['gMSMK','gMSMK mod1','gMSMK mod2'],
    dodge = True, 
    errwidth = 1.5, capsize = 0.1)

    y_max = df_ps[f].loc[df_ps["Feed Rate Fraction at Induction"]>=0.8].max() * 1.4
    axs[i].set(ylim = [0, y_max])

    plt.tight_layout()
    if i == 0:
        axs[i].legend(title = "Production Media", frameon = False, framealpha = 0,
        loc = "upper left", ncol = 1)
    else:
        axs[i].legend([], [])

plt.savefig(output_path / f"Antibiotic.png", facecolor='w')
plt.show()

print(df_ps.loc[df_ps["Feed Rate Fraction at Induction"]>=0.8].groupby(["Antibiotic","Production Media"])[features].describe())


#%% trend plots
df_trends_melt = df_trends.melt(id_vars= ["USP","Time (h)"],
value_vars =process_params,
value_name="Value",var_name="Parameter")
# ["OU Total", "dOUR/dt (2h MA)"]
# process_params
df_trends_melt.sort_values(by = ["USP", "Parameter", "Time (h)"],
ascending = True, inplace = True)

plt.style.use('default')
plt.rcParams.update({'font.size': 16})
g = sns.FacetGrid(df_trends_melt, col="Parameter", hue="USP",
col_wrap = 3, sharey = False, size = 4, aspect = 2)

g.map_dataframe(sns.lineplot, x="Time (h)", y="Value")
ref_lines_dict = {
    "Batch":0,
    "Uninduced":10,
    "Induced":24
    }
max_ys = df_trends_melt.groupby("Parameter")["Value"].max()
min_ys = df_trends_melt.groupby("Parameter")["Value"].min()
for i, ax in enumerate(g.axes):

    label_position_y = max_ys[i]*1.05
    ax.set_ylim([min_ys[i], max_ys[i]*1.15])

    for stage, start_time in ref_lines_dict.items():
        
        ax.axvline(start_time,
            ls = "--", linewidth = 1, alpha = 0.5, color = "black",
            label = stage)
        
        if i < 3:
            ax.text(start_time+0.5, label_position_y, stage, size=14) # horizontalalignment='left', verticalalignment='center', 
        # text_position_x += 10
g.add_legend(bbox_to_anchor=(1, 0.5), frameon=False, ncol = 1)
#g.add_legend(loc = "best",frameon=False, ncol = 2)
g.set_ylabels("")
g.set_titles(col_template="{col_name}")
g.tight_layout()
#g.savefig(output_path / "FermTrends_OU.png", facecolor='w')
g.savefig(output_path / "FermTrends_all.png", facecolor='w')
plt.show()

#%% Re

df_trends["Stirring Speed (RPS)"] = df_trends["Tip Speed (m/s)"] * np.pi * df_trends["Impeller Diameter (m)"]
df_trends["Stirring Speed (RPM)"] = df_trends["Stirring Speed (RPS)"] / 60
df_trends["Reynolds"] = reynolds_mixing(df_trends["Impeller Diameter (m)"], df_trends["Stirring Speed (RPS)"], rho = 1e3, mu = 8.9e-4)
df_trends["Regime"] = df_trends["Reynolds"].apply(lambda x: reynolds_chech(x))

lam_trans = 2000
trans_turb = 4000
turb = df_trends["Reynolds"].max() * 1.2
trange = df_trends["Time (h)"].min(), df_trends["Time (h)"].max()

plt.rcParams.update({'font.size': 16})
fig, ax = plt.subplots(figsize = (14,7))
sns.lineplot(ax = ax, data = df_trends, x = "Time (h)", y = "Reynolds", style = "Scale (l)")
ax.fill_between(trange, [lam_trans, lam_trans], [0,0], label = "Laminar", color = "grey", alpha = 0.5)
ax.fill_between(trange, [trans_turb, trans_turb], [lam_trans, lam_trans], label = "Transitional", color = "grey", alpha = 0.2)
ax.fill_between(trange, [turb, turb], [trans_turb, trans_turb], label = "Turbulent", color = "grey", alpha = 0.)

label_position_y = turb
for stage, start_time in ref_lines_dict.items():
    
    ax.axvline(start_time,
        ls = "--", linewidth = 1.5, alpha = 0.5, color = "black",
        label = None)
    ax.text(start_time+0.5, label_position_y, stage, size=14)

ax.set(yscale = "log", ylabel = r"$Re$")
ax.grid(which = "both", axis = "both")
plt.legend(title = "Scale (l)", loc="upper left", frameon = False, ncol = 1, 
            bbox_to_anchor=(1., .9), borderaxespad=0.)
fig.tight_layout()
plt.savefig(output_path/"ReynoldsNumber.png", facecolor='w')
plt.show()

#%% biomass density from q
df_trends_clean["Predicted Biomass Density (g/l)"] = (1/q0)*(df_trends_clean["Power (W/m3)"]**n_exp)

sns_lineplot(df_trends_clean, "Time (h)", "Predicted Biomass Density (g/l)", "Biomass Density (g/l)",
style_label = None, legend_var = True, save_fname = None,
save_path = output_path)


# %% plot features
def Scatter3D(x_vals, y_vals, z_vals,
x_label, y_label, z_label,
c_val = None, c_label = None, save_fname = None, save_path = None):
 
    fig = plt.figure(figsize = (5,5)) # dpi = 600
    ax = fig.add_subplot(projection = "3d")

    if c_label:

        im = ax.scatter(x_vals,
                    y_vals,
                    z_vals, c = c_val, 
                cmap = "cividis", marker = 'o',
                s = 100, alpha = 0.8, edgecolor = "black", label = c_label)
        
        fig.colorbar(im, ax=ax, label = f"{c_label}")
    else:
        im = ax.scatter(x_vals,
                    y_vals,
                    z_vals,  
                marker = 'o',
                s = 100, alpha = 0.5, edgecolor = jm_colors_list[0], color = jm_colors_list[0])
    ax.view_init(20, 30)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_zlabel(z_label)
    ax.tick_params(labelsize=12)
    #plt.tight_layout()
    
    if save_path and save_fname:
        image_name = save_path / (save_fname.replace(" ", "")+".png")
        plt.savefig(image_name, facecolor='w')
    
    plt.show()

def Scatter3D_cat(dataFrame, x_label, y_label, z_label, c_label, 
    save_fname = None, save_path = None):
 
    fig = plt.figure(figsize = (7,7)) # dpi = 600
    ax = fig.add_subplot(projection = "3d")

    m_list = ["o", "D"]
    for i, c_val in enumerate(dataFrame[c_label].unique()):
        dataFrame_plot = dataFrame.loc[dataFrame[c_label] == c_val]

        ax.scatter(dataFrame_plot[x_label],dataFrame_plot[y_label],dataFrame_plot[z_label],
            color = tab10_colors_list[i], marker = marker_list[i],
            s = 180, alpha = 0.4, edgecolor = tab10_colors_list[i], label = c_val)

    ax.view_init(20, 30)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_zlabel(z_label)
    ax.tick_params(labelsize=12)
    # plt.legend(title = c_label, loc = "upper left",
    #     bbox_to_anchor = (1.,0.8), frameon = False)

    plt.legend(title = c_label, loc = "center right",
    frameon = True, framealpha = 1)

    # fig.tight_layout()
    
    if save_path and save_fname:
        image_name = save_path / (save_fname.replace(" ", "")+".png")
        plt.savefig(image_name, facecolor='w')
    
    plt.show()

def scatter_2D(x_vals, y_vals, x_label, y_label, xlim = None, c_val = None, c_label = None,
save_fname = None, save_path = None):
    
    if c_label:
        fig, ax = plt.subplots(figsize = (8,6))
        edgecolor_var = "black"

        im = ax.scatter(x_vals, y_vals, c = c_val,
            cmap = "viridis_r", marker = 'o', s = 100, alpha = 0.7,
            edgecolor = edgecolor_var, label = c_label)
            
        fig.colorbar(im, ax=ax, label = f"{c_label}")
    else:
        fig, ax = plt.subplots(figsize = (6,6))
        edgecolor_var = "black"#None

        im = ax.scatter(x_vals, y_vals, marker = 'o', s = 100, alpha = 0.7,
            edgecolor = edgecolor_var)
    
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)

    plt.xlim(xlim) 
    plt.tight_layout()

    plt.grid(which = "both", axis = "both")
    
    if save_path and save_fname:
        image_name = save_path / (save_fname.replace(" ","") + ".png")
        
        plt.savefig(image_name, facecolor='w')
     
    plt.show()

def get_r2(x_vals, y_vals):
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x_vals, y_vals)
    return round(r_value**2,2)

plt.rcParams.update({'font.size': 14})

# Scatter3D(df[features[0]], df[features[1]], df[features[2]],
#     features[0], features[1], features[2],
#     save_fname = "Response3DScatter", save_path= output_path)

Scatter3D_cat(df_ps, features[0], features[1], features[2], "Scale (l)", 
    save_fname = "Response3DScatter", save_path= output_path)

plt.rcParams.update({'font.size': 14})

#%% acid tend by biomass den
sns_lineplot(df_trends_clean, "Time (h)", "Acid Total", "Biomass Density (g/l)",
style_label = None, palette_var = "Reds", legend_var = True, save_fname = "BiomassDensity_AcidTrend",
save_path = output_path)

#%% specific power differential
# from scipy import integrate

# df_trends_clean.sort_values(by = ["USP","Time (h)"], inplace = True, ascending = True)
# n = 1#0.47
# dt = 0.25
# df_trends_clean["OUR (mol/l/h)"] = df_trends_clean["Power (W/m3)"] ** n

# df_trends_clean["OU"] = df_trends_clean["OUR (mol/l/h)"] * dt # (mmol) ?
# df_trends_clean["Total OU"] = df_trends_clean.groupby(["USP"])["OU"].cumsum()

# df_trends_clean["dOUR"] = df_trends_clean.groupby(["USP"])["OUR (mol/l/h)"].diff(periods = 1)
# df_trends_clean["dOUR"].fillna(0, inplace = True)
# df_trends_clean["dOUR/dt"] = df_trends_clean["dOUR"]/dt

# df_trends_clean[["OUR (mol/l/h)"]] = MinMaxScaler().fit_transform(df_trends_clean[["OUR (mol/l/h)"]])

# sns_lineplot(df_trends_clean, "Time (h)", "Power (W/m3)", "Biomass Density (g/l)")
# sns_lineplot(df_trends_clean, "Time (h)", "OUR (mol/l/h)", "Biomass Density (g/l)")
# sns_lineplot(df_trends_clean, "Time (h)", "OU", "Biomass Density (g/l)")
# sns_lineplot(df_trends_clean, "Time (h)", "dOUR/dt", "Biomass Density (g/l)")

# def scale_col(series, scaler_var):
#%%



#%%

plt.rcParams.update({'font.size': 16})

fig, ax = plt.subplots(figsize = (12,7))
sns.lineplot(ax = ax, data = df_trends_clean,
    x = "Time (h)", y = "dOUR/dt (2h MA)",
    hue = "USP")#, palette = "viridis_r")#"Biomass Density (g/l)"
# ax.fill_between(trange, [lam_trans, lam_trans], [0,0], label = "Laminar", color = "grey", alpha = 0.5)
# ax.fill_between(trange, [trans_turb, trans_turb], [lam_trans, lam_trans], label = "Transitional", color = "grey", alpha = 0.2)
# ax.fill_between(trange, [turb, turb], [trans_turb, trans_turb], label = "Turbulent", color = "grey", alpha = 0.)

label_position_y = df_trends_clean["dOUR/dt (2h MA)"].max()*1.2
for stage, start_time in ref_lines_dict.items():
    
    ax.axvline(start_time,
        ls = "--", linewidth = 1, alpha = 0.5, color = "black",
        label = None)
    ax.text(start_time+0.5, label_position_y, stage, size=14)

y_lim = None#[1, label_position_y*1.5]
ax.set(yscale = "linear", ylabel = "$dOUR/dt$", ylim = y_lim)
ax.grid(which = "both", axis = "y")
# plt.legend(title = "USP", loc="best", frameon = True, framealpha = 1) # bbox_to_anchor=(1., .9), borderaxespad=0.
#plt.legend(loc = "best", ncol = 10)
fig.tight_layout()
plt.savefig(output_path/"Profile_dOURdt.png", facecolor='w')
plt.show()

#%%
plt.rcParams.update({'font.size': 16})

fig, ax = plt.subplots(figsize = (12,7))
sns.lineplot(ax = ax, data = df_trends_clean,
    x = "Time (h)", y = "OU Total",
    hue = "USP")#, palette = "viridis_r")#"Biomass Density (g/l)"
# ax.fill_between(trange, [lam_trans, lam_trans], [0,0], label = "Laminar", color = "grey", alpha = 0.5)
# ax.fill_between(trange, [trans_turb, trans_turb], [lam_trans, lam_trans], label = "Transitional", color = "grey", alpha = 0.2)
# ax.fill_between(trange, [turb, turb], [trans_turb, trans_turb], label = "Turbulent", color = "grey", alpha = 0.)

label_position_y = df_trends_clean["OU Total"].max()*1.
for stage, start_time in ref_lines_dict.items():
    
    ax.axvline(start_time,
        ls = "--", linewidth = 1, alpha = 0.5, color = "black",
        label = None)
    ax.text(start_time+0.5, label_position_y, stage, size=14)

y_lim = None#[1, label_position_y*1.5]
ax.set(yscale = "linear", ylabel = "$OU_{T}$", ylim = y_lim)
ax.grid(which = "both", axis = "y")
# plt.legend(title = "USP", loc="best", frameon = True, framealpha = 1) # bbox_to_anchor=(1., .9), borderaxespad=0.
plt.legend([],[], frameon=False)
fig.tight_layout()
plt.savefig(output_path/"Profile_OUTotal.png", facecolor='w')
plt.show()

#%% r2
import scipy
feature_combs = [(features[0],features[1]),(features[0],features[2]),(features[1],features[2])]

for combs in feature_combs:
    df_temp = df_ps[list(combs)].dropna(axis = 0)
    print(combs, " ", get_r2(df_temp[combs[0]], df_temp[combs[1]]))

print("Done")

# %%
