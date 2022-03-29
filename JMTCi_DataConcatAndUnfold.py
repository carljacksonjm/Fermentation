"""
Ingests production summary report and pairs with the trends datasets
Feature engineering
Unfold data and export
Data viz
"""
#%% imports
from JMTCi_Functions import reynolds_mixing, DoPickle, tip_speed, P_per_V, froude_mixing
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

#%% functions
def day_counter(row_val):
    global day_count

    if row_val < 0:
        day_count += 1
    
    return day_count

def clean_nasty_floats(nasty_series):
    good_series = nasty_series.replace("\t","", regex = True)
    good_series = good_series.replace(" ","", regex = True)

    return pd.to_numeric(good_series)

def remove_date(row_str):
    if " " in row_str:
        row_str = str(row_str).split(" ")[-1]
    return str(row_str)

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

#%% input variable
source_path = Path("C:/Users/JacksC01/OneDrive - Johnson Matthey/Documents/DigitalModelling/JMTCi4Fermentation/data/trends")
summary_path = Path("C:/Users/JacksC01/OneDrive - Johnson Matthey/Documents/DigitalModelling/JMTCi4Fermentation/data")

output_path = Path("C:/Users/JacksC01/OneDrive - Johnson Matthey/Documents/DigitalModelling/JMTCi4Fermentation/figures")

processed_path = summary_path

summary_fname = "Production summary for Carl.xlsx"
output_fname1 = "trends_processed.csv"
output_fname2 = "obswise_fermentation.csv"
output_fname3 = "batchwise_fermentation.csv"
# output_fname3 = "batchwise_fermentation_trimmed.csv"

impeller_d_dict = {
    1: 0.038, 42: 0.089,
    0.85: 0.038, 27.5: 0.089
    }
vol_scale_dict = {
    1: 0.85, 42: 27.5
    }
n_impeller_dict = {
    1: 2, 42: 3
    }

vessel_d_dict = { 
    1: 0.090, 42: 0.267
    }


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

dt = 0.25
# spec_power_exponent = 0.33 #0.47
# spec_power_intercept = 11.249370177870125 # A*ug**beta*dCO2
t_rolling = 2 # hours

time_all = list(np.arange(0.0, 46, 1))
time_batch = list(np.arange(0.0, 10, 1))#list(np.arange(0.0, 45.5, 0.5))
time_induction = list(np.arange(-2, 22, 1))
time_feed = list(np.arange(-2., 15, 1))#list(np.arange(0.0, 14.5, 0.5))#[0.5, 2, 12]

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
df_ps["Vessel Diameter (m)"] = df_ps["Scale"].map(vessel_d_dict)

df_ps["Feed Media"] = df_ps["Feed Media and sequence"].apply(lambda x: x.split("_")[0])
df_ps["Feed Sequence"] = df_ps["Feed Media and sequence"].apply(lambda x: x.split("_")[1])

df_ps["Feed Reduction at Induction (%)"] = df_ps["Feed Sequence"].map(seq_red_dict)
df_ps["Feed Rate Fraction at Induction"] = (100 - df_ps["Feed Reduction at Induction (%)"])/100

df_ps["Stepped Feed Sequence"] = df_ps["Feed Sequence"].map(seq_type_dict)
df_ps["Biomass Density (g/l)"] = df_ps['Total Biomass (g)'] / df_ps['Volume (l)']

ps_x_vars = ["Feed Media", "Glycerol ID","Production Media","Scale",
"Induction Temp (°C)","Induction pH", "Antibiotic",
"Feed Rate Fraction at Induction", "Stepped Feed Sequence",'Induction x_bar OD600, Aus']

y_cols = ['Volumetric Activity (U/mL)', 'Harvest x_bar WCW (g/L)',
"Biomass Density (g/l)"]

df_ps_clean = df_ps[ps_x_vars + y_cols]
dummy_cols = df_ps_clean.select_dtypes(include = ["object"]).columns.to_list()
df_ps_clean = pd.get_dummies(df_ps_clean, prefix = dummy_cols, prefix_sep = "_")

dupe_cols =  ['Glycerol ID_WGS-0020','Antibiotic_None', "Stepped Feed Sequence_Ramp"]
df_ps_clean.drop(columns = dupe_cols, inplace = True)

# X_cols_ps = list(set(df_ps_clean.columns) - set(y_cols))

print(df_ps_clean.columns.to_list())
# total acid and base
df_nans = df_ps.groupby(['Scale','Volume (l)',"DOI (dd/mm/yr)",'USP'])[['25% ammonia used (ml)','2M H3PO4 used (ml)','Vessel Diameter (m)','Impeller Diameter (m)',"Number of Impellers"]].mean()

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
    right = df_trends_usp[["ACID_PUMP_RATE", "BASE_PUMP_RATE", "Scale", 'Volume (l)',"Impeller Diameter (m)","Vessel Diameter (m)","Number of Impellers"]],
    how = "left",left_on = "USP",right_index = True)
df_trends["ACID_TOTAL_CALC"] = (df_trends["ACID_PUMP"] * df_trends["ACID_PUMP_RATE"])
df_trends["BASE_TOTAL_CALC"] = (df_trends["BASE_PUMP"] * df_trends["BASE_PUMP_RATE"])
df_trends["ACID_TOTAL"] = df_trends["ACID_TOTAL_CALC"] / df_trends["Volume (l)"]
df_trends["BASE_TOTAL"] = df_trends["BASE_TOTAL_CALC"] / df_trends["Volume (l)"]
df_trends['ANTI_FOAM'] = df_trends.groupby(["USP"])['ANTI_FOAM'].cumsum()
df_trends[["ACID_TOTAL","BASE_TOTAL"]] = df_trends[["ACID_TOTAL","BASE_TOTAL"]].fillna(value = 0) 
df_trends["FEED_FRACTION"] = df_trends.groupby(["USP"])["FEED_PUMP"].apply(lambda x: x / x.max())

# scale params
df_trends["Stirring Speed (RPS)"] = df_trends['STIRRER']/60
df_trends["Tip Speed (m/s)"] = tip_speed(df_trends["Impeller Diameter (m)"], df_trends["Stirring Speed (RPS)"])
df_trends["Power (W/m3)"] = P_per_V(df_trends["Stirring Speed (RPS)"],df_trends["Impeller Diameter (m)"],df_trends["Volume (l)"]/1000, 5, n_imps = df_trends["Number of Impellers"])
df_trends["log(Re)"] = np.log10(reynolds_mixing(df_trends["Impeller Diameter (m)"],df_trends["Stirring Speed (RPS)"]))
df_trends["log(Fr)"] = np.log10(froude_mixing(df_trends["Stirring Speed (RPS)"],df_trends["Impeller Diameter (m)"]))

df_trends.sort_values(by = ["USP","BATCH_TIME_H"], inplace = True, ascending = True)

# df_trends["OUR"] = df_trends["Power (W/m3)"] ** spec_power_exponent #spec_power_intercept * 
# df_trends["OU"] = df_trends["OUR"] * dt # (mmol) ?
# df_trends["OU Total"] = df_trends.groupby(["USP"])["OU"].cumsum()
df_trends["E (J/m3)"] = df_trends["Power (W/m3)"] * dt
df_trends["log(E Total)"] = np.log10(df_trends.groupby(["USP"])["E (J/m3)"].cumsum())
df_trends['log(Power)'] = np.log10(df_trends["Power (W/m3)"])
# df_trends[df_trends["PH"] < 6] = np.nan
# df_trends.dropna(how = "any", subset = ["PH"], axis = 0, inplace = True)
#  'log(Power)',
#  "log(E Total)",
#  "log(Re)",
#  "log(Fr)":

# df_trends["dOUR"] = df_trends.groupby(["USP"])["OUR"].diff(periods = 1)
# df_trends["dOUR"].fillna(0, inplace = True)
# df_trends["dOUR/dt"] = df_trends["dOUR"]/dt

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

moving_av_cols = ["ACID_TOTAL_DIFF","BASE_TOTAL_DIFF"] # ,"dOUR/dt"
for col in moving_av_cols:
    new_col = col + "_MA"
    grouped_moving_av = df_trends.groupby('USP')[col].rolling(av_window, min_periods=1).mean()
    df_trends[new_col] = grouped_moving_av.reset_index(level=0, drop=True)
    print(new_col)

scale_cols = ['ACID_TOTAL', "ACID_TOTAL_DIFF_MA", 'BASE_TOTAL', "BASE_TOTAL_DIFF_MA"]#,
   # "OUR","OU Total"] # "dOUR/dt_MA",
df_trends[scale_cols] = MinMaxScaler().fit_transform(df_trends[scale_cols].to_numpy())

#%%
df_trends.drop(columns = ["FEED_PUMP","Power (W/m3)",#"OU",#"dOUR/dt",
    "ACID_PUMP","ACID_TOTAL_CALC", "ACID_PUMP_RATE",
    "BASE_PUMP","BASE_TOTAL_CALC", "BASE_PUMP_RATE",
    'AF_PUMP',"FILE_TYPE",'BASE_PUMP','FEED_PUMP','GAS_MIX',
    "ACID_TOTAL_DIFF","BASE_TOTAL_DIFF","ANTI_FOAM","Number of Impellers",
    "Stirring Speed (RPS)", 'Impeller Diameter (m)', 'Vessel Diameter (m)',"E (J/m3)"], inplace = True)

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
    # "dOUR/dt_MA":"dOUR/dt (2h MA)",
    # "OU Total":"OU Total",
    # "OUR":"OUR",
#  'log(Power)',
#  "log(E Total)",
#  "log(Re)",
#  "log(Fr)":
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
    'Protein Activity (U/mg)':'Protein Activity (U/mg)',
    'Induction x_bar OD600, Aus':'Induction OD600'
    }

df_trends.rename(columns = rename_dict, inplace = True)
df_trends = df_trends.fillna(0)#.drop(columns = bad_cols)
#df_all = df_trends.copy()

df_ps_clean.rename(columns = rename_dict, inplace = True)
df_ps.rename(columns = rename_dict, inplace = True) 
y_cols = ['Biomass Density (g/l)', 'Harvest WCW (g/l)', 'Volumetric Activity (U/ml)']

process_params = ['Temperature (°C)',
 'pH',
 'DO2 (%)',
 'Acid Total',
 'Base Total',
 'Feed Total',
 'Tip Speed (m/s)',
 'log(Power)',
 "log(E Total)",
 'Acid (2h MA)',
 'Base (2h MA)',
 "log(Re)",
 "log(Fr)"
]

time_invar_params = list(set(df_ps_clean.columns) - set(y_cols + process_params))# + ["Scale"]

#%% observation-wise unfolding
df_trends_clean = pd.merge(left = df_trends[["USP","Time (h)"]+process_params],
    right = df_ps_clean[time_invar_params + y_cols],
how = "left",
left_on = "USP", right_index = True)

print(f"File observation-wise unfolding\nShape {df_trends_clean.shape}\n---")
df_trends_clean.to_csv(processed_path / output_fname2)

#%% batch-wise unfolding data from batch start, feed start, induction start
# batch start
df_uf = df_trends[process_params + ["USP", "Time (h)"]]
df_uf.set_index(["USP","Time (h)"], drop = True, inplace = True)
df_uf = df_uf[df_uf.index.get_level_values('Time (h)').isin(time_all)]#.unstack()

df_uf_t = df_uf[df_uf.index.get_level_values('Time (h)').isin(time_batch)].unstack()
df_uf_t.columns = [f"{col[0]}, tb={col[1]}" for col in df_uf_t.columns.to_list()]

#%% feed start
df_uf_feed = df_uf.copy()
df_uf_feed.reset_index(drop = False, inplace = True)

s_batch_end = df_uf_feed[df_uf_feed["Feed Total"]>0.025].groupby("USP")["Time (h)"].min()
s_batch_end.name = "Batch End Time (h)"
time_invar_params.append("Batch End Time (h)")

df_uf_feed = pd.merge(df_uf_feed, s_batch_end, left_on = "USP", right_index=True, how = "left")
df_uf_t = pd.merge(df_uf_t, s_batch_end, left_index = True, right_index=True, how = "left")

df_uf_feed['Feeding Time (h)'] = df_uf_feed["Time (h)"] - df_uf_feed["Batch End Time (h)"]
df_uf_feed.set_index(["USP","Feeding Time (h)"], inplace = True, drop = True)
df_uf_feed = df_uf_feed[process_params]

df_uf_feed = df_uf_feed[df_uf_feed.index.get_level_values('Feeding Time (h)').isin(time_feed)].unstack()
df_uf_feed.dropna(how = "any", axis = 1, inplace = True)
df_uf_feed.columns = [f"{col[0]}, tf={col[1]}" for col in df_uf_feed.columns.to_list()]

#%% induction start
df_uf_i = df_uf.copy()
df_uf_i["Induction Time (h)"] = df_uf_i.index.get_level_values('Time (h)') - 24
df_uf_i.reset_index(drop = False, inplace= True)
df_uf_i.set_index(["USP", "Induction Time (h)"], drop = True ,inplace = True)
df_uf_i = df_uf_i[process_params]
df_uf_i = df_uf_i[df_uf_i.index.get_level_values("Induction Time (h)").isin(time_induction)].unstack()
df_uf_i.columns = [f"{col[0]}, ti={col[1]}" for col in df_uf_i.columns.to_list()]

#%%
df_all = df_uf_t
for df_merging in [df_uf_feed, df_uf_i, df_ps_clean]:
    df_all = pd.merge(left = df_all, right = df_merging, how = "left", left_index=True, right_index=True)

print("Unfolded dataframe shape: ", df_all.shape)

X_cols = list(set(df_all.columns.to_list()) - set(y_cols))
time_var_params = list(set(X_cols) - set(time_invar_params)) 
print("Trend data joined with Production Summary data.")


# check for nulls
print("Columns with nulls", df_all.columns[df_all.isna().any()].tolist())
print(f"File batch-wise unfolding data\nShape {df_all.shape}\n---")
df_all.to_csv(processed_path / output_fname3)

# exporting columns
all_cols_dict = {
    "ProcessParams":process_params,
    "y_cols":y_cols,
    "TimeVarParams":time_var_params,
    "TimeInvarParams":time_invar_params
}

DoPickle("AllColsDict.pkl").pickle_save(all_cols_dict)

#%%
print("Done.")

#%%