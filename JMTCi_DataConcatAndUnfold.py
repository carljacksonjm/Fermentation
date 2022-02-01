"""
base pump , contin / discrete 
some media better for growth than others?
optimal pcas for each stage
pc values for each stage
"""
#%% imports
from JMTCi_Functions import *
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

#%% input variable
source_path = Path("C:/Users/JacksC01/OneDrive - Johnson Matthey/Documents/DigitalModelling/JMTCi4Fermentation/data/trends")
summary_path = Path("C:/Users/JacksC01/OneDrive - Johnson Matthey/Documents/DigitalModelling/JMTCi4Fermentation/data")
output_path = Path("C:/Users/JacksC01/OneDrive - Johnson Matthey/Documents/DigitalModelling/JMTCi4Fermentation/figures")

processed_path = summary_path

summary_fname = "Production summary for Carl.xlsx"
output_fname1 = "trends_processed.csv"
output_fname2 = "obswise_fermentation.csv"
output_fname3 = "batchwise_fermentation.csv"

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

dt = 0.25
spec_power_exponent = 0.33 #0.47
spec_power_intercept = 11.249370177870125 # A*ug**beta*dCO2
t_rolling = 2 # hours

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

y_cols = ['Volumetric Activity (U/mL)', 'Harvest x_bar WCW (g/L)',
"Biomass Density (g/l)"]

df_ps_clean = df_ps[ps_x_vars + y_cols]
dummy_cols = df_ps_clean.select_dtypes(include = ["object"]).columns.to_list()
df_ps_clean = pd.get_dummies(df_ps_clean, prefix = dummy_cols, prefix_sep = "_")

dupe_cols =  ['Glycerol ID_WGS-0020','Antibiotic_None', "Stepped Feed Sequence_Ramp"]
df_ps_clean.drop(columns = dupe_cols, inplace = True)

X_cols_ps = list(set(df_ps_clean.columns) - set(y_cols))
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

df_trends["Stirring Speed (RPS)"] = df_trends["Tip Speed (m/s)"] * np.pi * df_trends["Impeller Diameter (m)"]
df_trends["Re"] = reynolds_mixing(df_trends["Impeller Diameter (m)"], df_trends["Stirring Speed (RPS)"], rho = 1e3, mu = 8.9e-4)


#
from scipy import integrate
df_trends.sort_values(by = ["USP","BATCH_TIME_H"], inplace = True, ascending = True)

df_trends["OUR"] = spec_power_intercept * df_trends["Power (W/m3)"] ** spec_power_exponent 
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
    "ACID_TOTAL_DIFF","BASE_TOTAL_DIFF","ANTI_FOAM","Number of Impellers",
    "Stirring Speed (RPS)", 'Impeller Diameter (m)'], inplace = True)

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

y_cols = ['Volumetric Activity (U/ml)','Harvest WCW (g/l)',
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
 "dOUR/dt (2h MA)","Re"
]

time_invar_params = list(set(df_ps_clean.columns) - set(y_cols + process_params))# + ["Scale"]

#%% observation-wise unfolding
df_trends_clean = pd.merge(left = df_trends[["USP","Time (h)"]+process_params],
    right = df_ps_clean[time_invar_params + y_cols],
how = "left",
left_on = "USP", right_index = True)

print(f"File observation-wise unfolding\nShape {df_trends_clean.shape}\n---")
df_trends_clean.to_csv(processed_path / output_fname2)

#%% batch-wise unfolding data from batch start
df_uf = df_trends[process_params + ["USP", "Time (h)"]]
df_uf.set_index(["USP","Time (h)"], drop = True, inplace = True)

batch_t = [2, 7, 27, 37, 45] #12, 22, 27, 35, 45
df_uf_t = df_uf[df_uf.index.get_level_values('Time (h)').isin(batch_t)].unstack()
df_uf_t.columns = [f"{col[0]}, t={col[1]}" for col in df_uf_t.columns.to_list()]

#%% batch-wise unfolding data from feed start
df_uf["Feeding Time (h)"] = np.where(df_uf["Feed Total"]>0.025, df_uf.index.get_level_values('Time (h)'), np.nan)
df_uf_feed = df_uf.dropna(how = "any", subset = ["Feeding Time (h)"])

df_uf_feed.reset_index(drop = False, inplace = True)
df_uf_feed.drop(columns = "Time (h)", inplace = True)

df_uf_t["Batch End (h)"] = df_uf_feed.groupby("USP")["Feeding Time (h)"].min()

df_uf_feed["Feeding Time (h)"] = df_uf_feed.groupby("USP")["Feeding Time (h)"].apply(lambda x: x - x.min())
df_uf_feed.set_index(["USP","Feeding Time (h)"], inplace = True, drop = True)

batch_feed = [0.5, 2, 12]
df_uf_feed = df_uf_feed[df_uf_feed.index.get_level_values('Feeding Time (h)').isin(batch_feed)].unstack()
df_uf_feed.columns = [f"{col[0]}, tf={col[1]}" for col in df_uf_feed.columns.to_list()]

#%%
df_all = pd.merge(left = df_uf_t, right = df_uf_feed,
how = "left",
left_index = True, right_index = True)
df_all = pd.merge(left = df_all, right = df_ps_clean,
how = "left",
left_index = True, right_index = True)
print("Unfolded dataframe shape: ", df_all.shape)

X_cols = list(set(df_all.columns.to_list()) - set(y_cols))
print("Trend data joined with Production Summary data.")

# check for nulls
print("Columns with nulls", df_all.columns[df_all.isna().any()].tolist())
print(f"File batch-wise unfolding data\nShape {df_all.shape}\n---")
df_all.to_csv(processed_path / output_fname3)

print("Done.")

#%%
df_fedbatch = df_trends_clean.loc[df_trends_clean["Feed Total"]>0.025]
df_fedbatch.reset_index(drop = True, inplace = True)
df_fedbatch["Feeding Time (h)"] = df_fedbatch.groupby("USP")["Time (h)"].apply(lambda x: x - x.min())
df_fedbatch = df_fedbatch.loc[df_fedbatch["Feeding Time (h)"]<15]

#%% Feed total during fed-batch stage
plt.rcParams.update({'font.size': 16})
fig, ax = plt.subplots(figsize = (12,7))
sns.lineplot(ax = ax, data = df_fedbatch,
    x = "Feeding Time (h)", y = "Feed Total",
    hue = "Harvest WCW (g/l)", palette = "viridis_r")#"Biomass Density (g/l)"

ax.axvspan(1.5, 2.5, alpha=0.3, color='grey')

ax.set(yscale = "linear", ylabel = "Feed Fraction Total", xlabel = "Feeding Time (h)")
ax.grid(which = "both", axis = "y")
# plt.legend(title = "USP", loc="best", frameon = True, framealpha = 1) # bbox_to_anchor=(1., .9), borderaxespad=0.
plt.legend([],[], frameon=False)
fig.tight_layout()
plt.savefig(output_path/"HarvestWCW_FeedTotalTrend.png", facecolor='w')
plt.show()

#%% OU total during fed-batch stage
fig, ax = plt.subplots(figsize = (12,7))
sns.lineplot(ax = ax, data = df_fedbatch,
    x = "Feeding Time (h)", y = "OU Total",
    hue = "Harvest WCW (g/l)", palette = "viridis_r")

ax.axvspan(11.5, 12.5, alpha=0.3, color='grey')
ax.set(yscale = "linear", ylabel = "OU Total", xlabel = "Feeding Time (h)")
ax.grid(which = "both", axis = "y")
plt.legend(title = "Harvest WCW (g/l)", loc="best", frameon = True, framealpha = 1)
fig.tight_layout()
plt.savefig(output_path/"HarvestWCW_OUTotalTrend.png", facecolor='w')
plt.show()

#%% base plot during batch phase
fig, ax = plt.subplots(figsize = (12,7))
sns.lineplot(ax = ax, data = df_trends_clean.loc[df_trends_clean["Time (h)"]<10],
    x = "Time (h)", y = "Base (2h MA)",
    hue = "Harvest WCW (g/l)", palette = "viridis_r")#"Biomass Density (g/l)"

ax.axvspan(6.5, 7.5, alpha=0.3, color='grey')

ax.set(yscale = "linear", ylabel = "Base (2h MA)", xlabel = "Time (h)")
ax.grid(which = "both", axis = "y")
plt.legend(title = "Harvest WCW (g/l)", loc="best", frameon = True, framealpha = 1)
fig.tight_layout()
#plt.savefig(output_path/"HarvestWCW_OUTotalTrend.png", facecolor='w')
plt.show()

#%% volumetric activity specific power trend
fig, ax = plt.subplots(figsize = (12,7))
sns.lineplot(ax = ax, data = df_trends_clean.loc[df_trends_clean["Time (h)"]<10],
    x = "Time (h)", y = "Power (W/m3)",
    hue = "Volumetric Activity (U/ml)", palette = "viridis_r")

ax.axvspan(6.5, 7.5, alpha=0.3, color='grey')

ax.set(yscale = "linear", ylabel = "Power (W/m$^{3}$)", xlabel = "Time (h)")
ax.grid(which = "both", axis = "y")
plt.legend(title = "Volumetric Activity (U/ml)", loc="upper left", frameon = True, ncol = 2,framealpha = 1)
fig.tight_layout()
plt.savefig(output_path/"VolumetricActivity_Power.png", facecolor='w')
plt.show()

#%% feed rate fraction
fig, axs = plt.subplots(1,3, figsize = (18,6))

for i, f in enumerate(y_cols):

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

for i, f in enumerate(y_cols):

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

#%% all trend plots
df_trends_melt = df_trends.melt(id_vars= ["USP","Time (h)"],
value_vars =process_params,
value_name="Value",var_name="Parameter")

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
g.set_ylabels("")
g.set_titles(col_template="{col_name}")
g.tight_layout()
g.savefig(output_path / "FermTrends_all.png", facecolor='w')
plt.show()

#%% plot Re regimes
df_trends["Regime"] = df_trends["Re"].apply(lambda x: reynolds_chech(x))
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

#%% plot response variables
plt.rcParams.update({'font.size': 14})
Scatter3D_cat(df_ps, y_cols[0], y_cols[1], y_cols[2], "Scale (l)", 
    save_fname = "Response3DScatter", save_path= output_path)
plt.rcParams.update({'font.size': 14})

#%% acid tend by biomass density
sns_lineplot(df_trends_clean, "Time (h)", "Acid Total", "Biomass Density (g/l)",
style_label = None, palette_var = "Reds", legend_var = True, save_fname = "BiomassDensity_AcidTrend",
save_path = output_path)

#%%
print("Done")
