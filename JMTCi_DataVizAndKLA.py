"""
base pump , contin / discrete 
some media better for growth than others?
optimal pcas for each stage
pc values for each stage
"""
#%% imports
from locale import normalize
from itsdangerous import NoneAlgorithm
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from sympy import true
from JMTCi_Functions import DoPickle

import scipy
from matplotlib.ticker import FormatStrFormatter

import warnings
warnings.filterwarnings('ignore')

cb_colors_list = sns.color_palette("muted")+sns.color_palette("muted")  # deep, muted, pastel, bright, dark, and colorblind

plt.style.use('default')
plt.rcParams.update({'font.size': 16})

#%%
def get_r2(x_vals, y_vals):
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x_vals, y_vals)
    return slope, intercept, round(r_value**2,2)

def scatter_contin(
    dataFrame, x_col, y_col, hue_col, 
    x_label, y_label, hue_label,
    x_scale = "linear", y_scale = "linear", x_lim = None, y_lim = None,
    fig_size_var = (12,8),palette_var = "viridis_r", hue_lim = [0,1], alpha_var = 0.5, marker_var = "s",
    save_fname = None, save_path = None):
    
    fig, ax = plt.subplots(figsize = fig_size_var)
    
    m = ax.scatter(dataFrame[x_col], dataFrame[y_col], c = dataFrame[hue_col],
    cmap = palette_var, vmin = hue_lim[0], vmax = hue_lim[1], alpha = alpha_var, s = 100, marker = marker_var)

    ax.set(ylabel = y_label, xlabel = x_label, xscale = x_scale, yscale = y_scale,
        xlim = x_lim, ylim = y_lim)

    ax.grid(which = "major", axis = "both")

    fig.colorbar(m, label = hue_label,#cax = ax[len_col_list],
        fraction = .05, pad = 0.04, orientation = "vertical")      
    plt.tight_layout()
    
    if save_path and save_fname:
        image_name = save_path / (save_fname.replace(" ", "")+".png")
        plt.savefig(image_name, facecolor='w')
     
    plt.show()

def Scatter3D(x_vals, y_vals, z_vals,
x_label, y_label, z_label,
c_val = None, c_label = None, x_rot = 30, marker_size = 170, figsize_var = (11,5), palette_var = "viridis_r",alpha_var = 0.6,
x_lim = None,y_lim = None, z_lim = None,
save_fname = None, save_path = None):
 
    fig = plt.figure(figsize = figsize_var) # dpi = 600
    ax = fig.add_subplot(projection = "3d")

    if c_label:
        im = ax.scatter(x_vals,
                    y_vals,
                    z_vals, c = c_val, 
                cmap = palette_var, marker = 'o',
                s = marker_size, alpha = alpha_var, edgecolor = None, label = c_label)

        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
   
        fig.colorbar(im, ax = ax,label = f"{c_label}",
            fraction=0.025, pad=0.04)# ax=ax, 
    else:
        im = ax.scatter(x_vals,
                    y_vals,
                    z_vals,  
                marker = 'o', s = marker_size, alpha = alpha_var, linewidth = 2,
                edgecolor = cb_colors_list[0], color = cb_colors_list[0])
    ax.view_init(15, x_rot)
    ax.set(ylabel= y_label, xlabel=x_label, zlabel=z_label,
    xlim = x_lim, ylim = y_lim, zlim = z_lim)

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax.tick_params(labelsize=12)
    #fig.tight_layout()
    #ax.tight_layout()
    
    if save_path and save_fname:
        image_name = save_path / (save_fname.replace(" ", "")+".png")
        plt.savefig(image_name, facecolor='w')
    
    plt.show()

def multi_scatter_contin(
    dataFrame, x_col, y_col, hue_col, col_col,
    x_label, y_label, hue_label, col_label,
    x_scale = "linear", y_scale = "linear", x_lim = None, y_lim = None,
    fig_size_var = (12,8),palette_var = "viridis_r", hue_lim = [0,1], alpha_var = 0.5, marker_var = "s",
    save_fname = None, save_path = None):
    
    col_list = dataFrame[col_col].unique()
    len_col_list = len(col_list)

    plt_widths = [1] * (len_col_list+1)
    plt_widths[-1] = 0.05
    fig, ax = plt.subplots(1,len_col_list+1, figsize = fig_size_var, gridspec_kw={'width_ratios': plt_widths})
    
    
    for i, col in enumerate(col_list):

        dataFrame_temp = dataFrame.loc[dataFrame[col_col] == col]
    
        m = ax[i].scatter(dataFrame_temp[x_col], dataFrame_temp[y_col], c = dataFrame_temp[hue_col],
        cmap = palette_var, vmin = hue_lim[0], vmax = hue_lim[1], alpha = alpha_var, s = 100, marker = marker_var)

        ax[i].set(ylabel = "", xlabel = x_label, title = col, xscale = x_scale, yscale = y_scale,
        xlim = x_lim, ylim = y_lim)

        if i in range(1,len_col_list+1):
            ax[i].axes.yaxis.set_ticklabels([])

        if i in range(0,len_col_list+1):
            ax[i].grid(which = "major", axis = "both")

    ax[0].set(ylabel = y_label)
    fig.colorbar(m, label = hue_label, cax = ax[len_col_list],
        #fraction = .5, pad = 0.1, orientation = "vertical", aspect = 30)  
        fraction = .05, pad = 0.04, orientation = "vertical")      
    plt.tight_layout()
    
    if save_path and save_fname:
        image_name = save_path / (save_fname.replace(" ", "")+".png")
        plt.savefig(image_name, facecolor='w')
     
    plt.show()

#%% input variable
source_path = Path("C:/Users/JacksC01/OneDrive - Johnson Matthey/Documents/DigitalModelling/JMTCi4Fermentation/data")
output_path = Path("C:/Users/JacksC01/OneDrive - Johnson Matthey/Documents/DigitalModelling/JMTCi4Fermentation/figures")

df_trends_clean = pd.read_csv(source_path / "obswise_fermentation.csv")
df_all = pd.read_csv(source_path / "batchwise_fermentation.csv")

df_fermid = pd.read_excel(source_path / "Production summary for Carl.xlsx", sheet_name = "FermenterID")
df_fermid["USP"] = df_fermid['USP'].apply(lambda x: int(x.replace("USP","")))#.astype(int)
df_all = df_all.merge(df_fermid, left_on = "USP", right_on = "USP")
df_all.set_index("USP", drop = True, inplace = True)

df_trends_clean = df_trends_clean.merge(df_fermid, left_on = "USP", right_on = "USP")

#%%
all_cols_dict = DoPickle("AllColsDict.pkl").pickle_load()

y_cols = all_cols_dict['y_cols']
X_cols_tvar = all_cols_dict['TimeVarParams']
X_cols_tinvar = all_cols_dict['TimeInvarParams']
X_cols = X_cols_tinvar + X_cols_tvar
process_params = all_cols_dict["ProcessParams"]

#%%
df_trends_clean["Production Media"] = np.where(df_trends_clean["GMSMK Prod Media"] == 1, "Unmod", 'Mod1')
df_trends_clean["Production Media"] = np.where(df_trends_clean['GMSMK Mod2 Prod Media'] == 1, "Mod2", df_trends_clean["Production Media"])
df_all["Production Media"] = np.where(df_all["GMSMK Prod Media"] == 1, "Unmod", 'Mod1')
df_all["Production Media"] = np.where(df_all['GMSMK Mod2 Prod Media'] == 1, "Mod2", df_all["Production Media"])

#%% all trend plots
df_trends_melt = df_trends_clean.melt(id_vars= ["USP","Time (h)"],
value_vars =process_params,
value_name="Value",var_name="Parameter")
df_trends_melt.sort_values(by = ["USP", "Parameter", "Time (h)"],
ascending = True, inplace = True)

plt.style.use('default')
plt.rcParams.update({'font.size': 16})

g = sns.FacetGrid(df_trends_melt, col="Parameter", hue="USP",
col_wrap = 3, sharey = False, size = 4, aspect = 2)
g.map_dataframe(sns.lineplot, x="Time (h)", y="Value")

max_ys = df_trends_melt.groupby("Parameter")["Value"].max()

for i, ax in enumerate(g.axes):

    label_position_y = max_ys[i]*1.05
    ax.set_ylim([0, max_ys[i]*1.15])
    ax.grid(which = "major", axis = "x")

    ax.axvspan(9,14, alpha=0.2, color='grey')
    ax.axvspan(14, 24, alpha=0.3, color='grey')
    ax.axvspan(24,45, alpha=0.1, color='blue')

    ax.axvline(9, ls = "--", linewidth = 1.2, alpha = 0.7, color = "black")
    ax.axvline(14, ls = "--", linewidth = 1.2, alpha = 0.7, color = "black")
        
    if i < 3:
        ax.text(0+0.25, 1.05, "Batch", size=14)
        ax.text(9+0.25, 1.05, "UFB", size=14)
        ax.text(24+0.25, 1.05, "IFB", size=14)

g.add_legend(bbox_to_anchor=(0.5, 0.12),
    loc = "center",frameon=False, ncol = 6)
g.set_ylabels("")
g.set_titles(col_template="{col_name}")
g.tight_layout()
g.savefig(output_path / "FermTrends_all.png", facecolor='w')
plt.show()

#%% plot Re regimes
# df_trends["Regime"] = df_trends["Re"].apply(lambda x: reynolds_check(x))
# lam_trans = 2000
# trans_turb = 4000
# turb = df_trends["Reynolds"].max() * 1.2
# trange = df_trends["Time (h)"].min(), df_trends["Time (h)"].max()

# plt.rcParams.update({'font.size': 16})
# fig, ax = plt.subplots(figsize = (14,7))
# sns.lineplot(ax = ax, data = df_trends, x = "Time (h)", y = "Reynolds", style = "Scale (l)")
# ax.fill_between(trange, [lam_trans, lam_trans], [0,0], label = "Laminar", color = "grey", alpha = 0.5)
# ax.fill_between(trange, [trans_turb, trans_turb], [lam_trans, lam_trans], label = "Transitional", color = "grey", alpha = 0.2)
# ax.fill_between(trange, [turb, turb], [trans_turb, trans_turb], label = "Turbulent", color = "grey", alpha = 0.)

# label_position_y = turb
# for stage, start_time in ref_lines_dict.items():
    
#     ax.axvline(start_time,
#         ls = "--", linewidth = 1.5, alpha = 0.5, color = "black",
#         label = None)
#     ax.text(start_time+0.5, label_position_y, stage, size=14)

# ax.set(yscale = "log", ylabel = r"$Re$")
# ax.grid(which = "both", axis = "both")
# plt.legend(title = "Scale (l)", loc="upper left", frameon = False, ncol = 1, 
#             bbox_to_anchor=(1., .9), borderaxespad=0.)
# fig.tight_layout()
# plt.savefig(output_path/"ReynoldsNumber.png", facecolor='w')
# plt.show()

# #%% plot response variables
# plt.rcParams.update({'font.size': 14})
# Scatter3D_cat(df_ps, y_cols[0], y_cols[1], y_cols[2], "Scale (l)", 
#     save_fname = "Response3DScatter", save_path= output_path)
# plt.rcParams.update({'font.size': 14})

#%% Volumetric Activity (U/ml)
# line plot of Base (2h MA)
plt.rcParams.update({'font.size': 14})
fig, ax = plt.subplots(1,2, figsize = (10,4), sharey = True,
    gridspec_kw={
        'width_ratios': [0.33,0.66]
        }
        )

sns.lineplot(ax = ax[0], data = df_trends_clean.loc[df_trends_clean["Time (h)"]<=3],
    x = "Time (h)", y = "Base (2h MA)",
    hue = "Volumetric Activity (U/ml)", palette = "mako_r", legend = False)
sns.lineplot(ax = ax[1], data = df_trends_clean.loc[df_trends_clean["Time (h)"]<45],
    x = "Time (h)", y = "Base (2h MA)",
    hue = "Volumetric Activity (U/ml)", palette = "mako_r", legend = True)#"Biomass Density (g/l)"

# feed stage
ax[1].axvspan(9,14, alpha=0.3, color='grey')
ax[1].axvline(9, ls = "--", linewidth = 1, alpha = 0.5, color = "black")
ax[1].axvline(14, ls = "--", linewidth = 1, alpha = 0.5, color = "black")
ax[1].text(9+0.25, 1.05, "Feed Start", size=14)

# induction stage
ax[1].axvline(24,ls = "--", linewidth = 1, alpha = 0.5, color = "black")
ax[1].text(24+0.25, 1.05, "Induction", size=14)

for single_ax in ax:
    # model sample point
    single_ax.axvspan(0,2, alpha=0.3, color='green')
    single_ax.text(0.+0.25, 1.05, r'$t_{b}$=1 h', size=14)

    single_ax.set(yscale = "linear", ylabel = "Base (2h MA)", xlabel = "Time (h)", ylim = [-0.1,1.2])
    single_ax.grid(which = "both", axis = "y")

plt.legend(title = "Activity (U/ml)",
    loc="upper left", frameon = False, ncol = 1, framealpha = 1,
    bbox_to_anchor=(1., 1.), borderaxespad=0.)

fig.tight_layout()

image_name = output_path / ("VolumetricActivityBaseTrend.png")
plt.savefig(image_name, facecolor='w')
plt.show()

# scatter plot of Base (2h MA) and Prod Media
plt.rcParams.update({'font.size': 14})
fig, ax = plt.subplots(figsize = (6,6))
markers_plot = ["^","o", "s"]

ax.scatter(x = df_all['Base (2h MA), tb=1.0'].loc[df_all['GMSMK Prod Media']==1], 
    y = df_all['Volumetric Activity (U/ml)'].loc[df_all['GMSMK Prod Media']==1],
    marker = markers_plot[0],
    color = cb_colors_list[2],  edgecolor = cb_colors_list[2], 
    label = 'gMSMK',
    s = 100, linewidth = 2, alpha = 0.5)

ax.scatter(x = df_all['Base (2h MA), tb=1.0'].loc[df_all['GMSMK Mod1 Prod Media']==1], 
    y = df_all['Volumetric Activity (U/ml)'].loc[df_all['GMSMK Mod1 Prod Media']==1],
    marker = markers_plot[1], 
    color = cb_colors_list[0],  edgecolor = cb_colors_list[0],
    label = 'gMSMK Mod1',
    s = 100, linewidth = 2, alpha = 0.5)

ax.scatter(x = df_all['Base (2h MA), tb=1.0'].loc[df_all['GMSMK Mod2 Prod Media']==1], 
    y = df_all['Volumetric Activity (U/ml)'].loc[df_all['GMSMK Mod2 Prod Media']==1],
    marker = markers_plot[2],
    color = cb_colors_list[1],  edgecolor = cb_colors_list[1],
    label = 'gMSMK Mod2',
    s = 100, linewidth = 2, alpha = 1)

ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.set(xlabel = r'Base (2h MA), $t_{b}$ = 1 h', ylabel = 'Volumetric Activity (U/ml)')
# plt.grid(which = "both", axis = "both")  xlim = [0, 0.7], ylim = [0,7]

plt.legend(title = "Production Media", loc="best",
    fancybox = False, framealpha = 1, edgecolor = "black",shadow = False) # 

plt.tight_layout()
image_name = output_path / ("VolumetricActivityBaseProdMedia.png")
plt.savefig(image_name, facecolor='w')
plt.show()

#%% scatter of harvest and activity, production media and temperature
fig, ax = plt.subplots(1,1,figsize = (10,6))

df_temp = df_all.copy()
df_temp.rename(columns = {'Induction Temperature (°C)':'Temperature (°C)'}, inplace = True)

sns.scatterplot(ax = ax, data = df_temp, y = "Harvest WCW (g/l)", x = "Volumetric Activity (U/ml)", style = 'Temperature (°C)',hue = "Production Media", 
style_order = [30,25], hue_order = ["Unmod", "Mod1", "Mod2"],
palette = "colorblind", markers = ["o","^"], s = 80, alpha = 0.8, edgecolor ="black")

ax.legend(bbox_to_anchor = (1,1),
    loc = "upper left", ncol = 1,
    fancybox = False, framealpha = 1, edgecolor = "black",shadow = False)

plt.tight_layout()
image_name = output_path / ("VolumetricActivityHarvestWCWProdMediaTempScatter.png")
plt.savefig(image_name, facecolor='w')
plt.show()

#%% Error pH f(Fr, Re, Activity)
df_plot = df_trends_clean[['Induction Temperature (°C)',"log(Fr)", "log(Re)","Base (2h MA)","Production Media",
"GMSMK Prod Media",'GMSMK Mod2 Prod Media','GMSMK Mod1 Prod Media','Feed Rate Fraction at Induction',
    "Time (h)","Scale (l)", "Tip Speed (m/s)","pH","Volumetric Activity (U/ml)", "FermenterID",'log(Power)','Temperature (°C)']]#.loc[df_trends_clean["Time (h)"]<5]

df_plot["Power"] = 10**df_plot['log(Power)']
df_plot['kla'] = df_plot['Power']**0.33
df_plot["Fr"] = 10** df_plot["log(Fr)"]
df_plot["Re"] = 10** df_plot["log(Re)"]
df_plot["Error pH"] = (6.7 - df_plot["pH"])
df_plot["Error pH (abs)"] = np.abs(df_plot["Error pH"])
df_plot["Production Media"] = pd.Categorical(df_plot["Production Media"], ["Unmod","Mod1", "Mod2"])
df_plot.sort_values(by = ["Production Media","Feed Rate Fraction at Induction","Time (h)"], inplace = True)
# df_plot.sort_values(by = "Volumetric Activity (U/ml)", inplace = True, ascending = False)
time_range = list(np.arange(1,48,1))
plt.rcParams.update({'font.size': 20})

#%%
multi_scatter_contin(df_plot.loc[(df_plot["Time (h)"].isin(time_range))],
    'Base (2h MA)', "Error pH (abs)", "Feed Rate Fraction at Induction", "Production Media",
    'Base (2h MA)', r"$|E_{pH}|$", "FRFI", "Media",
    y_scale = "log", x_scale = "linear", x_lim = [0.,1.1], y_lim = [0.0001,1.5],
    fig_size_var = (14,8),palette_var = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True), hue_lim = [0.5, 1], alpha_var = 0.6, marker_var = "s",
    save_fname = "FRFI_BasePHErrorMedia", save_path = output_path)

scatter_contin(df_plot.loc[(df_plot['Induction Temperature (°C)']>22) & (df_plot["Time (h)"]<12) & (df_plot["pH"]>4)],
    'Fr', "Error pH (abs)", 'Time (h)',
    'Fr', r"$|E_{pH}|$", 'Time (h)',
    y_scale = "log", x_scale = "linear",  x_lim = None, y_lim = None,
    fig_size_var = (12,8),
    palette_var = sns.diverging_palette(20, 220, l=65, center="dark", as_cmap=True),
    hue_lim = [0, 10], alpha_var = 0.5, marker_var = "s",
    save_fname = "Time_FrPHErrorMedia", save_path = output_path)

multi_scatter_contin(
    df_plot.loc[(df_plot['Induction Temperature (°C)']>22)&(df_plot["Time (h)"].isin(time_range))],
    "kla",'Base (2h MA)', "Volumetric Activity (U/ml)", "Production Media",
     r"$P_{v}^{\alpha}$",'Base (2h MA)', "Activity (U/ml)", "Media",
    y_scale = "linear", x_scale = "linear",  y_lim = [0,1.1], x_lim = [0,25],
    fig_size_var = (12,8),
    palette_var = "viridis_r",
    hue_lim = [2, 7], alpha_var = 0.5, marker_var = "s",
    save_fname = "Activity_BaseKlaMedia", save_path = output_path)

#%%
multi_scatter_contin(df_plot.loc[df_plot['Induction Temperature (°C)']>28],
    'Re',"Error pH (abs)", "Volumetric Activity (U/ml)", "Production Media",
    r'Re', r"$|E_{pH}|$", "Activity (U/ml)", "Production Media",
    y_scale = "log", x_scale = "log", x_lim = None, y_lim = None,
    fig_size_var = (12,8),palette_var = "mako_r", hue_lim = [0,7], alpha_var = 0.5, marker_var = "s",
    save_fname = None, save_path = None)

multi_scatter_contin(df_plot.loc[df_plot['Induction Temperature (°C)']>26],
    'kla','Base (2h MA)', "Volumetric Activity (U/ml)", "Production Media",
    r'$P_{v}^{\alpha}$', 'Base (2h MA)', "Activity (U/ml)", "Production Media",
    y_scale = "linear", x_scale = "linear", x_lim = None, y_lim = None,
    fig_size_var = (12,8),palette_var = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True), hue_lim = [1,7], alpha_var = 0.5, marker_var = "s",
    save_fname = "Activity_klaBaseMedia", save_path = None)

multi_scatter_contin(df_plot,#.loc[(df_plot["Scale (l)"]==1)],
    'kla', "Base (2h MA)", "Volumetric Activity (U/ml)", "Production Media",
    'kla', "Base (2h MA)", "Volumetric Activity (U/ml)", "Production Media",
    y_scale = "linear", x_scale = "linear", x_lim = None, y_lim = None,
    fig_size_var = (12,8),palette_var = "mako_r", hue_lim = [1,7], alpha_var = 0.5, marker_var = "s",
    save_fname = None, save_path = None)

multi_scatter_contin(df_plot.loc[(df_plot["Scale (l)"]==1)],
    "Fr", "Error pH (abs)", "Volumetric Activity (U/ml)", "Production Media",
    r"$Fr$", r"$|E_{pH}|$", "Activity (U/ml)", "Media",
    y_scale = "log", x_scale = "log", x_lim = [2e-2, 3e0], y_lim = [1e-5, 1e1],
    fig_size_var = (12,8),palette_var = "mako_r", hue_lim = [1,7], alpha_var = 0.5, marker_var = "s",
    save_fname = None, save_path = None)

#%%
scatter_contin(df_plot.loc[(df_plot['Induction Temperature (°C)']>22)],
    'Fr', "Base (2h MA)", "Feed Rate Fraction at Induction",
    'Fr', "Base (2h MA)", "FRFI",
    y_scale = "linear", x_scale = "linear",  x_lim = None, y_lim = None,
    fig_size_var = (12,8),
    palette_var = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True), hue_lim = [0.5, 1], alpha_var = 0.5, marker_var = "s",
    save_fname = "FRFI_FrBase", save_path = output_path)

#%%
df_temp = df_trends_clean.loc[(df_trends_clean["Time (h)"]>=0) & (df_trends_clean["Time (h)"]<3) & (df_plot["pH"]>-5)].groupby(["Scale (l)","USP"])[["pH","Volumetric Activity (U/ml)","Time (h)"]].min()

fig, ax = plt.subplots()
sns.lineplot(
    ax = ax,
    data = df_trends_clean.loc[(df_trends_clean["Time (h)"]>=0) & (df_trends_clean["Time (h)"]<2) & (df_plot["pH"]>-5)],
    y = "pH", x = "Time (h)", hue = "Volumetric Activity (U/ml)",
    legend = False, palette = "mako_r")
ax.set(xscale = "linear")
plt.show()

#%%
df_temp = df_trends_clean[["Volumetric Activity (U/ml)", "Scale (l)", "Base (2h MA)","Feed Total"]].copy()
df_temp["Activity (U/ml)"] = df_temp["Volumetric Activity (U/ml)"]

plt.rcParams.update({'font.size': 20})
fig, ax = plt.subplots(1,1, figsize = (16,6))

sns.lineplot(data = df_temp,
    x = "Feed Total", y = "Base (2h MA)", hue = "Activity (U/ml)", hue_norm = (2.,6),style = "Scale (l)",
    linewidth = 2, palette = "viridis_r", legend = "brief")

ax.set(yscale = "linear", ylabel = "Base (2h MA)", xlabel = r"$X_{F}$", ylim = [0.,1.], xlim = [0,1.])
ax.grid(which = "both", axis = "x")

ax.legend(bbox_to_anchor=(1, 1),
loc="upper left", ncol = 1,
fancybox = False, framealpha = 1, edgecolor = "black",shadow = False)

fig.tight_layout()
image_name = output_path / ("Activity_FeedBaseTrend.png")
plt.savefig(image_name, facecolor='w')
plt.show()


#%%
df_temp = df_plot.loc[(df_plot['Induction Temperature (°C)']>26)&(df_plot["Time (h)"]>1)]# &(df_plot["Time (h)"].isin(time_range))
# 'Base (2h MA)' , "Error pH (abs)"

Scatter3D(df_temp["Volumetric Activity (U/ml)"], df_temp['Fr'], df_temp["Error pH (abs)"] , 
"Activity (U/ml)","Fr", r"$|E_{pH}|$", 
c_val = df_temp['Base (2h MA)'], c_label = 'Base (2h MA)', x_rot = 60, marker_size = 120, figsize_var = (10,16), palette_var = "Reds", alpha_var = 1.,
x_lim = None, y_lim = [0,2.5], z_lim = [0,0.3],
save_fname = None, save_path = None)

#%% fermenter id and pH error
plt.rcParams.update({'font.size': 20})

fig, ax = plt.subplots(figsize = (12,6))

sns.barplot(ax = ax, data = df_plot.sort_values(by = "FermenterID"), dodge = True, orient = "horizontal", ci = None,
    y = "FermenterID", x = "Error pH (abs)", hue = "Production Media", alpha = 0.75,#color = "grey",
    edgecolor = None, palette = "colorblind", hue_order = ["Unmod", "Mod1", "Mod2"])

sns.stripplot(ax = ax, data = df_plot.sort_values(by = "FermenterID"),dodge = True,
    y = "FermenterID", x = "Error pH (abs)",
    hue = "Production Media", color = "black",hue_order = ["Unmod", "Mod1", "Mod2"],
    alpha = 0.7)

ax.set(xlabel = r"$|E_{pH}|$", ylabel = "Fermenter", xscale = "log")
ax.grid(which = "major", axis = "x")

h, l = ax.get_legend_handles_labels()
ax.legend(
    handles = h[-3:], labels = l[-3:],
    title = "Production\nMedia", loc="upper left", bbox_to_anchor = (1,1),
    fancybox = False, framealpha = 1, edgecolor = "black",shadow = False
    )

plt.tight_layout()
image_name = output_path / ("FermIDPHError.png")
plt.savefig(image_name, facecolor='w')
plt.show()

# production media and ph control (excluding 42 L)
plt.rcParams.update({'font.size': 20})

fig, ax = plt.subplots(figsize = (12,6))

sns.kdeplot(
    ax = ax, data = df_plot.loc[(df_plot["Scale (l)"] != 42) & (df_plot["Error pH (abs)"] != 0.)], 
    x = "Error pH (abs)", hue = "Production Media",hue_order = ["Unmod", "Mod1", "Mod2"],
    linewidth = 2,
    hue_norm = True, log_scale = True,
    alpha = 0.75, palette = "colorblind")

ax.get_yaxis().set_visible(False)

ax.set(xlabel = r"$|E_{pH}|$")
ax.grid(which = "major", axis = "x")

# h, l = ax.get_legend_handles_labels()
# ax.legend(
#     handles = h[-3:], labels = l[-3:],
#     title = "Production\nMedia", loc="upper left", bbox_to_anchor = (1,1),
#     fancybox = False, framealpha = 1, edgecolor = "black",shadow = False
#     )

plt.tight_layout()
image_name = output_path / ("PHErrorDist.png")
plt.savefig(image_name, facecolor='w')
plt.show()


#%%
fig, ax = plt.subplots(1,1,figsize = (12,3))

sns.stripplot(ax = ax, data = df_all, y = "Production Media", x = "Volumetric Activity (U/ml)", hue = 'Induction Temperature (°C)',
    palette = "colorblind", jitter = 0, marker = "D", s = 10, alpha = 0.7, order = ["Unmod", "Mod1", "Mod2"], edgecolor ="black")
sns.boxplot(ax = ax, data = df_all, y = "Production Media", x = "Volumetric Activity (U/ml)", color = "white", order = ["Unmod", "Mod1", "Mod2"])

ax.grid(which = "both", axis = "both")
ax.set(ylabel = "")
ax.legend(title = 'Temperature (°C)',loc = "upper right",
    fancybox = False, framealpha = 1, edgecolor = "black",shadow = False)
plt.tight_layout()
image_name = output_path / ("VolumetricActivityProdMediaBoxPlot.png")
plt.savefig(image_name, facecolor='w')
plt.show()


sd_30 = df_all["Volumetric Activity (U/ml)"].loc[(df_all["Production Media"]=="Mod1") & (df_all['Induction Temperature (°C)']==30)].std()
mean_30 = df_all["Volumetric Activity (U/ml)"].loc[(df_all["Production Media"]=="Mod1") & (df_all['Induction Temperature (°C)']==30)].mean()
mean_25 = df_all["Volumetric Activity (U/ml)"].loc[(df_all["Production Media"]=="Mod1") & (df_all['Induction Temperature (°C)']==25)].mean()

print((mean_30 - mean_25 ) / sd_30)
#%%
g = sns.displot(data=df_trends_clean, x="Base (2h MA)", y = "Volumetric Activity (U/ml)", col="Production Media",
col_order = ["Unmod", 'Mod1', 'Mod2'],
palette = "colorblind", hue = "Production Media", legend = False)

g.set_titles("{col_name}")
g.tight_layout()

image_name = output_path / ("VolumetricActivityDistBarPlotBase2hMA.png")
plt.savefig(image_name, facecolor='w')
plt.show()

#%% Harvest WCW (g/l)

def sns_lineplot(dataFrame, x_label, y_label, color_label, style_label = None, y_scale = "linear", x_scale = "linear", legend_var = False, title_var = None, save_path = None):
    
    fig, ax = plt.subplots(figsize = (10,6))
    
    sns.lineplot(x=x_label, y=y_label, hue = color_label, style = style_label, data=dataFrame, ax=ax,
    legend = True, palette = "mako_r")

    plt.grid(axis = "both")

    if legend_var == True:
            plt.legend(title = color_label, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    else:
        plt.legend([],[], frameon=False)
        
    ax.set(ylabel = y_label, xlabel = x_label, yscale = y_scale, xscale = x_scale)
    
    if title_var:
        fig.suptitle(title_var)
        
    plt.tight_layout()
    
    if save_path:
        image_name = save_path / (title_var.replace(" ", "")+".png")
        plt.savefig(image_name, facecolor='w')
     
    plt.show()

print("Harvest WCW R2\nBase T: {0}\nFeed Rate Frac Induction {1}".format(
    get_r2(df_all['Base Total, ti=3.0'], df_all['Harvest WCW (g/l)'])[2],
    get_r2(df_all["Feed Rate Fraction at Induction"], df_all['Harvest WCW (g/l)'])[2]
    ))

# Scatter3D(df_all['log(Power), ti=3.0'], df_all["Feed Rate Fraction at Induction"], df_all['Harvest WCW (g/l)'],
# r'log(Power), $t_{i}$ = 3 h', "Induction Feed Rate Fraction", 'Harvest WCW (g/l)',
# c_val = df_all['Harvest WCW (g/l)'], c_label = 'Harvest WCW (g/l)', x_rot = 50, figsize_var=(8,8), 
# save_fname = "HarvestWCWFeedFracPower3D", save_path = output_path)

# Scatter3D(df_all['log(Power), ti=3.0'], df_all["Feed Rate Fraction at Induction"], df_all['Temperature (°C), ti=-1.0'],
# r'log(Power), $t_{i}$ = 3 h', "Induction Feed Rate Fraction", 'Temperature (°C), ti=-1.0',
# c_val = df_all['Harvest WCW (g/l)'], c_label = 'Harvest WCW (g/l)', x_rot = 50, figsize_var=(8,8), 
# save_fname = None, save_path = output_path)

#%%
def Scatter3D(x_vals, y_vals, z_vals,
x_label, y_label, z_label,
c_val = None, c_label = None, x_rot = 30, marker_size = 170, figsize_var = (11,5), palette_var = "viridis_r",
save_fname = None, save_path = None):
 
    fig = plt.figure(figsize = figsize_var) # dpi = 600
    ax = fig.add_subplot(projection = "3d")

    # def create_surface(x_minmax, y_minmax, z_minmax):

    #     X = np.arange(x_minmax[0], x_minmax[1], 0.1)
    #     Y = np.arange(y_minmax[0], y_minmax[1], 0.1)
    #     X, Y = np.meshgrid(X, Y)
    #     Z = np.arange(z_minmax[0], z_minmax[1], 0.1)

    #     return X, Y, Z

    if c_label:

        im = ax.scatter(x_vals,
                    y_vals,
                    z_vals, c = c_val, 
                cmap = palette_var, marker = 'o',linewidth = 0.1,
                s = marker_size, alpha = 0.7, edgecolor = "white", label = c_label)

        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
   
        fig.colorbar(im, ax = ax,label = f"{c_label}",
            fraction=0.025, pad=0.04)# ax=ax, 

        # X_range, Y_range, Z_range = create_surface(
        #     [min(x_vals), max(x_vals)],
        #     [0, 0],
        #     [min(z_vals), max(z_vals)]
        #     )

        # ax.plot_surface(X_range, Y_range, Z_range, linewidth=0, antialiased=False)

    else:
        im = ax.scatter(x_vals,
                    y_vals,
                    z_vals,  
                marker = 'o', s = marker_size, alpha = 0.8, linewidth = 2,
                edgecolor = cb_colors_list[0], color = cb_colors_list[0])
    ax.view_init(20, x_rot)
    ax.set(ylabel= y_label, xlabel=x_label, zlabel=z_label)

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    ax.tick_params(labelsize=10)
    #fig.tight_layout()
    #ax.tight_layout()
    
    if save_path and save_fname:
        image_name = save_path / (save_fname.replace(" ", "")+".png")
        plt.savefig(image_name, facecolor='w')
    
    plt.show()

Scatter3D(
    df_trends_clean['Feed Rate Fraction at Induction'].loc[df_trends_clean["Time (h)"]>2],
    df_trends_clean['Time (h)'].loc[df_trends_clean["Time (h)"]>2],
    df_trends_clean['log(Power)'].loc[df_trends_clean["Time (h)"]>2],
    'FRFI',r'$t_{b}$ (h)',r'$log(Power)$',
    c_val =df_trends_clean['Harvest WCW (g/l)'].loc[df_trends_clean["Time (h)"]>2],
    c_label = 'Harvest WCW (g/l)', marker_size = 60, x_rot = 50, figsize_var=(14,12), 
    save_fname = "HarvestWCWLogPowerFRFITrend3D", save_path = output_path
)

# fig = plt.figure(figsize = (10,10)) # dpi = 600
# ax = fig.add_subplot(projection = "3d")

# for i, media in enumerate(df_trends_clean["Production Media"].unique()):
#     for batch in df_trends_clean.USP.unique():

#         ax.plot(
#             df_trends_clean['Harvest WCW (g/l)'].loc[(df_trends_clean["USP"] == batch)&(df_trends_clean["Time (h)"]>2) & (df_trends_clean["Production Media"]==media)],
#             df_trends_clean['Time (h)'].loc[(df_trends_clean["USP"] == batch)&(df_trends_clean["Time (h)"]>2) & (df_trends_clean["Production Media"]==media)],
#             df_trends_clean['log(Power)'].loc[(df_trends_clean["USP"] == batch)&(df_trends_clean["Time (h)"]>2) & (df_trends_clean["Production Media"]==media)], 
#             color = cb_colors_list[i], 
#             linewidth = 2, label = media)

# ax.view_init(20, 50)
# ax.set(ylabel= "", xlabel="", zlabel="")

# ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# ax.tick_params(labelsize=10)
# plt.legend(title = "", loc = "best")
# #fig.tight_layout()
# #ax.tight_layout()
# # image_name = save_path / (save_fname.replace(" ", "")+".png")
# # plt.savefig(image_name, facecolor='w')
# plt.show()

#%%
df_trends_clean["Power"] = 10**df_trends_clean["log(Power)"]
y_max = 1e5

plt.rcParams.update({'font.size': 14})
fig, ax = plt.subplots(1,1, figsize = (10,4))

sns.lineplot(ax = ax, data = df_trends_clean.loc[df_trends_clean["Time (h)"]>1],
    x = "Time (h)", y = "Power",
    hue = "Harvest WCW (g/l)", palette = "viridis_r", legend = True)

# feed stage
ax.axvspan(9,14, alpha=0.3, color='grey')
ax.axvline(9, ls = "--", linewidth = 1, alpha = 0.5, color = "black")
ax.axvline(14, ls = "--", linewidth = 1, alpha = 0.5, color = "black")
ax.text(9+0.25, y_max/2, "Feed Start", size=14)

# induction stage
ax.axvline(24,ls = "--", linewidth = 1, alpha = 0.5, color = "black")
ax.text(24+0.25, y_max/2, "Induction", size=14)

# model sample point
ax.axvspan(26.5,27.5, alpha=0.3, color='green')
ax.text(26.5+0.25, y_max/5, r'$t_{i}$=3 h', size=14)

ax.set(yscale = "log", ylabel = r"Power (W/m$^{3}$)", xlabel = "Time (h)", ylim = [10, y_max])
ax.grid(which = "both", axis = "y")

plt.legend(title = "Harvest WCW (g/l)",
    loc="best", ncol = 2,
    fancybox = False, framealpha = 1, edgecolor = "black",shadow = False)

fig.tight_layout()

image_name = output_path / ("HarvestWCWPower.png")
plt.savefig(image_name, facecolor='w')
plt.show()

#%%
# plt.rcParams.update({'font.size': 14})

# fig, ax = plt.subplots(figsize = (8,6))
# markers_plot = ["o","s", "D","+"]

# for i, frfi in enumerate([1.0, 0.8, 0.75, 0.5]):

#     ax.scatter(x = df_all['log(Power), ti=3.0'].loc[df_all["Feed Rate Fraction at Induction"]==frfi], 
#         y = df_all["Harvest WCW (g/l)"].loc[df_all["Feed Rate Fraction at Induction"]==frfi],
#         marker = markers_plot[i],
#         color = cb_colors_list[i],  edgecolor = cb_colors_list[i], 
#         label = frfi, #facecolor = "none",
#         s = 50, alpha = 0.7, linewidth = 2)

# ax.set(xlabel = r'log(Power), $t_{i}$ = 3 h', ylabel = "Harvest WCW (g/l)")

# plt.legend(title = "FRFI", loc="upper left",
#     fancybox = False, framealpha = 1, edgecolor = "black",shadow = False) # 

# plt.tight_layout()
# # image_name = output_path / ("VolumetricActivityBaseProdMedia.png")
# # plt.savefig(image_name, facecolor='w')
# plt.show()

#%%
def scatter_2D(x_vals, y_vals, x_label, y_label, xlim = None, ylim = None, fig_size = (8,6), x_scale = "linear", c_val = None, c_label = None,
save_fname = None, save_path = None):
    
    fig, ax = plt.subplots(figsize = fig_size)

    if c_label:
        
        edgecolor_var = "black"
        im = ax.scatter(x_vals, y_vals, c = c_val,
            cmap = "viridis_r", marker = 'o', s = 100, alpha = 0.7,linewidth = 1.,
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

    plt.grid(which = "both", axis = "both")
    
    if save_path and save_fname:
        image_name = save_path / (save_fname.replace(" ","") + ".png")
        
        plt.savefig(image_name, facecolor='w')
     
    plt.show()

scatter_2D(df_all["Feed Rate Fraction at Induction"], df_all['log(Power), ti=3.0'],
"FRFI", r'$log(Power)$', xlim = [1.02,0.48], ylim = [1.7,4.3], fig_size = (8,6), x_scale = "linear",
c_val = df_all["Harvest WCW (g/l)"], c_label = "Harvest WCW (g/l)",
save_fname = "HarvestWCWLogPowerti3FRFI", save_path = output_path)

# plt.rcParams.update({'font.size': 14})

# fig, ax = plt.subplots(figsize = (8,6))
# markers_plot = ["o","s", "+","D"]

# # for i, frfi in enumerate(["Unmod","Mod1","Mod2"]):
    
# ax.scatter(y = df_all['log(Power), ti=3.0'], 
# x = df_all["Feed Rate Fraction at Induction"],
# c = df_all["Harvest WCW (g/l)"],
# s = 50, linewidth = 2, marker = markers_plot[0])

# # ax.set(xlabel = r'log(Power), $t_{i}$ = 3 h', ylabel = "Harvest WCW (g/l)")

# plt.legend(title = "Media", loc="upper left",
#     fancybox = False, framealpha = 1, edgecolor = "black",shadow = False) # 

# plt.tight_layout()
# # image_name = output_path / ("VolumetricActivityBaseProdMedia.png")
# # plt.savefig(image_name, facecolor='w')
# plt.show()

#%%


#%% PCA data prep
pca_path = create_dir("PCA", output_path)
X = df[X_cols_pcr].to_numpy()
scaler_X = StandardScaler()
scaler_X.fit(X) # get mean and std dev for scaling
X_scaled = scaler_X.transform(X) # scaled

usp_index = df.index.to_list()

y = df[features].to_numpy()
scaler_y = StandardScaler()
scaler_y.fit(y) # get mean and std dev for scaling
y_scaled = scaler_y.transform(y) # scaled
df_y_norm = pd.DataFrame(data = y_scaled,
    columns = features, index = usp_index)

print("Data ready.")

#%% viz
y = df[features].to_numpy()
scaler_y = MinMaxScaler()#MinMaxScaler()
scaler_y.fit(y) # get mean and std dev for scaling
y_scaled = scaler_y.transform(y) # scaled

X = df[X_cols].to_numpy()
X_cols = X_cols
scaler_X = MinMaxScaler()#MinMaxScaler()
scaler_X.fit(X) # get mean and std dev for scaling
X_scaled = scaler_X.transform(X) # scaled

#%% kla correlation
xi, ci, yi = [X_cols.index("Power (W/m3), t=45.0"),
    X_cols.index("Scale (l)"),
    features.index('Biomass Density (g/l)')]#

x_c_y = np.array([X[:, xi], X[:, ci], y[:, yi]]).T # rows, columns
x_c_y = x_c_y[~np.isnan(x_c_y).any(axis = 1), :] # drop infs
x_func = np.linspace(x_c_y[:,0].min(), x_c_y[:,0].max(), 50)

x_y_log = np.log10(x_c_y[:,[0,2]])
n_exp, log_A_exp, r_value, p_value, std_err = linregress(x_y_log[:,0], x_y_log[:, 1])
confidence_interval = 1.96*std_err
r2_exp = r_value ** 2
y_pred_log = n_exp * np.log10(x_func) + log_A_exp
y_conf = np.array([y_pred_log - confidence_interval, y_pred_log + confidence_interval]).T
A_exp = 10**log_A_exp
print(f"R2: {r2_exp: .2f}\nBD = (P/V)**{n_exp: .2f} + {A_exp: .2f}")

alpha_stats = 0.05
CI = [n_exp + std_err*t for t in stats.t.interval(alpha_stats/2, len(x_c_y[:,0])-2)]
halfwidth = std_err*stats.t.interval(alpha_stats/2, len(x_c_y[:,0])-2)[1]
n_exp_str = r'$\alpha$={:.2f}($\pm${:.3f})'.format(n_exp, halfwidth)

n_35 = 0.35 # Gill et al 2008
n_07 = 0.7 # Linek et al 2004, Van't Riet 1979
# kLa = 0.24 (PU/Vliq)0.7 (Ug)0.3 [Middleton Eq 15.17a in Edwards, Harnby & Nienow]
# kLa  (Pg/Vliq)0.7 (Ug)0.6, kla = 0.95 (P/V)**0.6 Ug**0.6

fig, ax = plt.subplots(figsize = (10,6))
m_list = ["o", "D"]

ax.plot(x_func, 10**(y_pred_log),
    linestyle = "-", color = "black", label = n_exp_str)

ax.scatter(x_c_y[x_c_y[:,1] == 1, 0], x_c_y[x_c_y[:,1] == 1,2],
        marker = m_list[1], s = 60, linewidth = 1.4,
        edgecolor = tab10_colors_list[0], label = "1 L", facecolor = "none")
ax.scatter(x_c_y[x_c_y[:,1] == 42, 0], x_c_y[x_c_y[:,1] == 42,2],
        marker = m_list[0], s = 60,linewidth = 1.4,
        edgecolor = tab10_colors_list[1], label = "42 L", facecolor = "none")

ax.fill_between(x_func, 10**(y_conf[:,1]), 10**y_conf[:, 0],
    color = "grey", alpha = 0.2, label = "95% CI")

for i, n_vals in enumerate([0.35, 0.7]):
    y_lit_log_max = n_vals * np.log10(x_func) + log_A_exp
    ax.plot(x_func, 10**(y_lit_log_max),
        linestyle = linestyle_list[i+1], color = tab10_colors_list[i+2], label = r"$\alpha$={:.2f}".format(n_vals))

ax.set(ylabel = "Biomass Density (g/l)", xlabel = "Power (W/m$^{3}$)", yscale = "log", xscale = "log")
# plt.legend(loc = "upper left", frameon = True, framealpha = 1)
plt.legend(loc="upper left", frameon = False, ncol = 1, 
            bbox_to_anchor=(1., 1.), borderaxespad=0.)
ax.grid(which = "both", axis = "both")

plt.tight_layout()
image_name = output_path / ("BiomassDensity_PowerScatter27KLA".replace(" ", "")+".png")
plt.savefig(image_name, facecolor='w')
plt.show()

#%% vol activity plots
x_vals = np.log10(X[:, X_cols.index("Power (W/m3), t=7.0")])
y_vals = np.log10(y[:, features.index('Volumetric Activity (U/ml)')])

q, q_int, std_error, r2_value, rmse_value, x_func, y_func = lin_reg(x_vals, y_vals)
confidence_interval = 1.96*std_error
y_conf = np.array([y_func - confidence_interval, y_func + confidence_interval]).T
print(f"R2 {r2_value:.2f}, RMSE {rmse_value: .2f}\nGrad = **{q:.2f}")

fig, ax = plt.subplots(figsize = (10,6))

ax.scatter(
    X[(X[:,X_cols.index("Scale (l)")] == 1) & (X[:,X_cols.index("Antifoam")] == 0), X_cols.index("Power (W/m3), t=7.0")],
    y[(X[:,X_cols.index("Scale (l)")] == 1) & (X[:,X_cols.index("Antifoam")] == 0), features.index('Volumetric Activity (U/ml)')],
    marker = marker_list[0], facecolor = "none", label = "1 l",
    color = tab10_colors_list[0], s = 150, alpha = 1, linewidth = 2)
ax.scatter(
    X[(X[:,X_cols.index("Scale (l)")] == 1) & (X[:,X_cols.index("Antifoam")] == 1), X_cols.index("Power (W/m3), t=7.0")],
    y[(X[:,X_cols.index("Scale (l)")] == 1) & (X[:,X_cols.index("Antifoam")] == 1), features.index('Volumetric Activity (U/ml)')],
    marker = marker_list[0], facecolor = "none", label = "1 l (A)",
    color = tab10_colors_list[1], s = 150, alpha = 1, linewidth = 2)

ax.scatter(
    X[(X[:,X_cols.index("Scale (l)")] == 42) & (X[:,X_cols.index("Antifoam")] == 0), X_cols.index("Power (W/m3), t=7.0")],
    y[(X[:,X_cols.index("Scale (l)")] == 42) & (X[:,X_cols.index("Antifoam")] == 0), features.index('Volumetric Activity (U/ml)')],
    marker = marker_list[1], facecolor = "none", label = "42 l",
    color = tab10_colors_list[0], s = 150, alpha = 1, linewidth = 2)

ax.scatter(
    X[(X[:,X_cols.index("Scale (l)")] == 42) & (X[:,X_cols.index("Antifoam")] == 1), X_cols.index("Power (W/m3), t=7.0")],
    y[(X[:,X_cols.index("Scale (l)")] == 42) & (X[:,X_cols.index("Antifoam")] == 1), features.index('Volumetric Activity (U/ml)')],
    marker = marker_list[1], facecolor = "none", label = "42 l (A)",
    color = tab10_colors_list[1], s = 150, alpha = 1, linewidth = 2)

ax.plot(10**x_func, 10**y_func, linestyle = "--", color = "black")
ax.fill_between(10**x_func, 10**y_conf[:,1], 10**y_conf[:, 0],
    color = "grey", alpha = 0.2, label = "95% CI")

plt.legend(loc="upper left", frameon = False, ncol = 1, 
            bbox_to_anchor=(1., 1.), borderaxespad=0.)
ax.grid(which = "both", axis = "both")

ax.set(xlabel = "Power, t=7.0 (W/m$^{3}$)", ylabel = "Volumetric Activity (U/ml)", xscale = "log", yscale ="log")
# ax.grid(which = "both", axis = "both")

plt.tight_layout()
image_name = output_path / ("VolumetricActivity_Power7".replace(" ", "")+".png")
plt.savefig(image_name, facecolor='w')
plt.show()

#%% harvest wcw plots
scatter_2D(X_scaled[:, X_cols.index("OU Total, tf=12.0")],y_scaled[:, features.index('Harvest WCW (g/l)')],
"OU Total, tf=12.0","Harvest WCW",
c_val = X_scaled[:, X_cols.index("Feed Total, tf=2.0")],
c_label = "Feed Total, tf=2.0",
save_fname = "HarvestWCW_FeedTotalOUTotal", save_path = output_path)

scatter_2D(X_scaled[:, X_cols.index("Feed Total, tf=12.0")], y_scaled[:, features.index('Harvest WCW (g/l)')],
"Feed Total, tf=12.0","Harvest WCW")

#save_fname = "HarvestWCW_FeedTotalOUTotal", save_path = output_path)


#%%

#%% specific oxygen uptake rate
df_plot = df.dropna(subset = ['Biomass Density (g/l)'])

q, q_int, r_value, p_value, std_err = linregress(df_plot['Biomass Density (g/l)'], df_plot['Power (W/m3), t=45.0']**n_exp)
x_func = np.linspace(df_plot['Biomass Density (g/l)'].min(), df_plot['Biomass Density (g/l)'].max(), 50)
y_func = x_func * q + q_int
print(q)
# scatter_2D(df['Power (W/m3), t=45.0']**n_exp, df['Biomass Density (g/l)'], "OUR",'Biomass Density (g/l)')

fig, ax = plt.subplots(figsize = (10,6))

ax.plot(x_func, y_func,
    linestyle = "--", color = "black", label = r"$\frac{dOTR}{dX}$="+f"{q:.2f}")

ax.scatter(df['Biomass Density (g/l)'], df['Power (W/m3), t=45.0']**n_exp, 
        s = 60, linewidth = 1.4, color = "black", facecolor = "none")

ax.set(xlabel = r"$X$ (g/l)", ylabel = r"$OTR$")
plt.legend(loc = "upper left", frameon = True, framealpha = 1)

plt.tight_layout()
image_name = output_path / ("specificO2ConsumptionRate".replace(" ", "")+".png")
plt.savefig(image_name, facecolor='w')
plt.show()

#%% BiomassDensityOUTotal
y_plot = y_scaled[~np.isnan(y[:,features.index('Biomass Density (g/l)')]),features.index('Biomass Density (g/l)')]
x_plot = X_scaled[~np.isnan(y[:,features.index('Biomass Density (g/l)')]), X_cols.index("OU Total, t=45.0")]

q, q_int, std_error, r2, rmse, x_range, y_range = lin_reg(x_plot, y_plot)
confidence_interval = 1.96*std_error
y_conf = np.array([y_range - confidence_interval, y_range + confidence_interval]).T
print(f"R2 {r2:.2f}, RMSE {rmse: .2f}")

fig, ax = plt.subplots(figsize = (10,6))

ax.plot(x_range, y_range,
    linestyle = "--", color = "black")# label = r"$BD =$ "+f"{q:.1f}"+r" $OU_{T}$ + " + f"{q_int:.1f}"

ax.scatter(x_plot, y_plot,
        s = 60, linewidth = 1.4, color = "black", facecolor = "none")

ax.fill_between(x_range, y_conf[:,1], y_conf[:, 0],
    color = "grey", alpha = 0.2, label = "95% CI")

ax.set(xlabel = "OU Total", ylabel = "Biomass Density")
plt.legend(loc = "upper left", frameon = True, framealpha = 1)

plt.tight_layout()
image_name = output_path / ("BiomassDensityOUTotal".replace(" ", "")+".png")
plt.savefig(image_name, facecolor='w')
plt.show()


#%%
fig, ax = plt.subplots(figsize = (10,6))

ax.scatter(
    X_scaled[X[:,X_cols.index('GMSMK Prod Media')] == 1,X_cols.index("OU Total, t=12.0")],
    y_scaled[X[:,X_cols.index('GMSMK Prod Media')] == 1,features.index('Volumetric Activity (U/ml)')],
    label = "gMSMK",
    marker = marker_list[0], #facecolor = "none",
    color = tab10_colors_list[0], edgecolor = tab10_colors_list[0], s = 150, alpha = 0.5, linewidth = 2)

ax.scatter(
    X_scaled[X[:,X_cols.index('GMSMK Mod1 Prod Media')] == 1,X_cols.index("OU Total, t=12.0")],
    y_scaled[X[:,X_cols.index('GMSMK Mod1 Prod Media')] == 1,features.index('Volumetric Activity (U/ml)')],
    label = "gMSMK Mod1",
    marker = marker_list[1], #facecolor = "none",
    color = tab10_colors_list[1], edgecolor = tab10_colors_list[1], s = 150, alpha = 0.5, linewidth = 2)

ax.scatter(
    X_scaled[X[:,X_cols.index('GMSMK Mod2 Prod Media')] == 1,X_cols.index("OU Total, t=12.0")],
    y_scaled[X[:,X_cols.index('GMSMK Mod2 Prod Media')] == 1,features.index('Volumetric Activity (U/ml)')],
    label = "gMSMK Mod2",
    marker = marker_list[2], #facecolor = "none",
    color = tab10_colors_list[2], edgecolor = tab10_colors_list[2], s = 150, alpha = 0.5, linewidth = 2)

ax.set(xlabel = "OU Total, t=12.0", ylabel = "Volumetric Activity")
# ax.grid(which = "both", axis = "both")

plt.legend(title = "Production Media",loc = "best",
    framealpha = 1, shadow = False)
plt.tight_layout()
image_name = output_path / ("VolActivity_OUTotalProdMedia".replace(" ", "")+".png")
plt.savefig(image_name, facecolor='w')
plt.show()

#%%
fig, ax = plt.subplots(figsize = (10,6))

ax.scatter(
    X_scaled[X[:,X_cols.index('GMSMK Prod Media')] == 1,X_cols.index("dOUR/dt (2h MA), t=27.0")],
    y_scaled[X[:,X_cols.index('GMSMK Prod Media')] == 1,features.index('Volumetric Activity (U/ml)')],
    label = "gMSMK",
    marker = marker_list[0], #facecolor = "none",
    color = tab10_colors_list[0], edgecolor = tab10_colors_list[0], s = 150, alpha = 0.5, linewidth = 2)

ax.scatter(
    X_scaled[X[:,X_cols.index('GMSMK Mod1 Prod Media')] == 1,X_cols.index("dOUR/dt (2h MA), t=27.0")],
    y_scaled[X[:,X_cols.index('GMSMK Mod1 Prod Media')] == 1,features.index('Volumetric Activity (U/ml)')],
    label = "gMSMK Mod1",
    marker = marker_list[1], #facecolor = "none",
    color = tab10_colors_list[1], edgecolor = tab10_colors_list[1], s = 150, alpha = 0.5, linewidth = 2)

ax.scatter(
    X_scaled[X[:,X_cols.index('GMSMK Mod2 Prod Media')] == 1,X_cols.index("dOUR/dt (2h MA), t=27.0")],
    y_scaled[X[:,X_cols.index('GMSMK Mod2 Prod Media')] == 1,features.index('Volumetric Activity (U/ml)')],
    label = "gMSMK Mod2",
    marker = marker_list[2], #facecolor = "none",
    color = tab10_colors_list[2], edgecolor = tab10_colors_list[2], s = 150, alpha = 0.5, linewidth = 2)

ax.set(xlabel = "dOUR/dt (2h MA), t=27.0", ylabel = "Volumetric Activity")
# ax.grid(which = "both", axis = "both")

plt.legend(title = "Production Media",loc = "best",
    framealpha = 1, shadow = False)
plt.tight_layout()
image_name = output_path / ("VolActivity_dOURdtProdMedia".replace(" ", "")+".png")
plt.savefig(image_name, facecolor='w')
plt.show()

#%%
df_X_scaled = pd.DataFrame(data = X_scaled,
columns = X_cols)
df_y_scaled = pd.DataFrame(data = y_scaled,
columns = features)

df_scaled = pd.merge(left = df_X_scaled, right = df_y_scaled, left_index = True, right_index = True)
df_scaled.index = df.index

#%%
basic_scatter(df, "dOUR/dt (2h MA), t=27.0", 'Volumetric Activity (U/ml)', z_string = "GMSMK Mod1 Prod Media")

basic_scatter(df, "Base (2h MA), t=22.0", 'Volumetric Activity (U/ml)', z_string = "GMSMK Mod1 Prod Media")

basic_scatter(df, "Base Total, t=22.0", 'Volumetric Activity (U/ml)', z_string = "GMSMK Mod1 Prod Media")

#%%
scatter_2D(df["OU Total, t=12.0"], df["Base (2h MA), t=22.0"], "x", "y",
    c_val = df['Volumetric Activity (U/ml)'], c_label = "Volumetric Activity (U/ml)")

scatter_2D(X_scaled[:,X_cols.index("OU Total, t=12.0")], y_scaled[:,features.index('Volumetric Activity (U/ml)')],
"OU Total, t=12.0","Volumetric Activity",
c_val = X_scaled[:,X_cols.index("Base (2h MA), t=22.0")], c_label = "Base (2h MA)")

scatter_2D(X_scaled[:,X_cols.index("Base (2h MA), t=22.0")], y_scaled[:,features.index('Volumetric Activity (U/ml)')],
"Base (2h MA), t=22.0","Volumetric Activity",
c_val = X_scaled[:,X_cols.index("dOUR/dt (2h MA), t=22.0")], c_label = "dOUR/dt (2h MA), t=22.0")

scatter_2D(df["dOUR/dt (2h MA), t=27.0"], df['Volumetric Activity (U/ml)'],"x", "y",
c_val = df["Base (2h MA), t=22.0"], c_label = "z")

# Scatter3D(X_scaled[:,X_cols.index("Power (W/m3), t=12.0")],
# X_scaled[:,X_cols.index('Base (2h MA), t=12.0')], y_scaled[:,features.index('Volumetric Activity (U/ml)')],
# "Specific Power","Base (2h MA)","Volumetric Activity", x_rot = 20,
# #c_val = X_scaled[:,X_cols.index("Temperature (°C), t=9.0")], c_label = "Temperature (°C), t=9.0",
# #c_val = y_scaled[:,features.index('Volumetric Activity (U/ml)')], c_label = 'Volumetric Activity',
# save_fname = "VolumetricActivity_BasePower12", save_path = output_path)

plt.rcParams.update({'font.size': 16})

print("Done")

#%% harvest wcw
# ['Base Total, t=4.0', 'PO2 (%), t=27.0', 'Acid (2h MA), t=4.0', 'PO2 (%), t=4.0',
# 'pH, t=27.0',
# 'OU Total, t=35.0']

print(get_r2(np.log10(X[:,X_cols.index("Power (W/m3), t=27.0")]), np.log10(y[:,features.index('Harvest WCW (g/l)')])))
print(get_r2((X[:,X_cols.index('OU Total, t=35.0')]),
(y[:,features.index('Harvest WCW (g/l)')])))

scatter_2D(X_scaled[:,X_cols.index('OU Total, t=35.0')],
y[:,features.index('Harvest WCW (g/l)')], "OU Total, t=35.0 h","Harvest WCW (g/l)",
c_val = X[:,X_cols.index('pH, t=27.0')], c_label = "pH, t=27.0 h",
save_fname = "HarvestWCW_OUTotalpH", save_path = output_path)
 # 10**(-1*X[:,X_cols.index('pH, t=27.0')])

scatter_2D(X[:,X_cols.index('pH, t=27.0')],
y[:,features.index('Harvest WCW (g/l)')], "pH, t=27.0 h","Harvest WCW (g/l)",
x_scale = "linear", fig_size = (10,6), save_fname = "HarvestWCW_pH", save_path = output_path)

scatter_2D(X[:,X_cols.index('OU Total, t=27.0')],
X[:,X_cols.index('pH, t=27.0')], "x","y",
x_scale = "linear", fig_size = (10,6))

#%%
fig, ax = plt.subplots(figsize = (8,8))
ax.scatter(
    X[X[:,X_cols.index('Production Media_gMSMK')] == 1,X_cols.index("Power (W/m3), t=27.0")],
    y[X[:,X_cols.index('Production Media_gMSMK')] == 1,features.index('Harvest WCW (g/l)')],
    label = "gMSMK",
    marker = marker_list[0], #facecolor = "none",
    color = tab10_colors_list[0], edgecolor = tab10_colors_list[0], s = 150, alpha = 0.5, linewidth = 2)

ax.scatter(
    X[X[:,X_cols.index('Production Media_gMSMK mod1')] == 1,X_cols.index("Power (W/m3), t=27.0")],
    y[X[:,X_cols.index('Production Media_gMSMK mod1')] == 1,features.index('Harvest WCW (g/l)')],
    label = "gMSMK mod1",
    marker = marker_list[1], #facecolor = "none",
    color = tab10_colors_list[1], edgecolor = tab10_colors_list[1],s = 150, alpha = 0.5, linewidth = 2)
ax.scatter(
    X[X[:,X_cols.index('Production Media_gMSMK mod2')] == 1,X_cols.index("Power (W/m3), t=27.0")],
    y[X[:,X_cols.index('Production Media_gMSMK mod2')] == 1,features.index('Harvest WCW (g/l)')],
    label = "gMSMK mod2",
    marker = marker_list[2], #facecolor = "none",
    color = tab10_colors_list[2], edgecolor = tab10_colors_list[2], s = 150, alpha = 0.5, linewidth = 2)

ax.set(xlabel = "Power (W/m$^{3}$)", ylabel = "Harvest WCW (g/l)", xscale = "log")
ax.grid(which = "both", axis = "both")
plt.legend(title = "Production Media",loc = "best",
    framealpha = 1, shadow = False)
plt.tight_layout()
image_name = output_path / ("HarvestWCW_PowerProdMediat27".replace(" ", "")+".png")
plt.savefig(image_name, facecolor='w')
plt.show()

# scatter_2D(X_latents[:,0], X_latents[:,1], x_label = "LV1", y_label = "LV2", xlim = [-2.5, 2.5], ylim = [-2.5, 2.5],
# c_val = y_pls, c_label = y_col_short, fig_size = (8,7), save_fname = f"{y_col_short}_LatentVars", save_path = y_col_path)


#%% plot loadings scatter
df_loadings = pd.DataFrame(data = pls_loadings, columns = ["PLS"+str(i+1) for i in range(len(pls_loadings.T))])
df_loadings["Parameter"] = X_cols_sfs
df_loadings_melt = pd.melt(df_loadings, id_vars = ["Parameter"], var_name = "Component", value_name = "Loading")
df_loadings_melt = df_loadings_melt.loc[df_loadings_melt["Loading"].abs() >0.05]
df_loadings["Time (h)"] = df_loadings["Parameter"].apply(lambda x: x.split(", t=")[-1])
df_loadings["Time (h)"] = df_loadings["Parameter"].apply(lambda x: x.split(", t=")[-1])
df_loadings = df_loadings.loc[(df_loadings["PLS1"].abs() >0.05) | (df_loadings["PLS2"].abs() >0.05)]
# df_loadings["Parameter"] = df_loadings["Parameter"].apply(lambda x: x.split(", t=")[0])

#%%
plt.rcParams.update({'font.size': 18})
fig, ax = plt.subplots(figsize = (10,6))
# sns.scatterplot(data = df_loadings, x = "PLS1", y = "PLS2",
#     hue = "Time (h)", style = "Parameter", palette = "Blues", s = 150)
sns.scatterplot(data = df_loadings, x = "PLS1", y = "PLS2",
    hue = "Parameter", style = "Parameter",palette = tab10_colors_list[:2], s = 150)
plt.grid(which = "both", axis = "both")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2,
    frameon = True, shadow = False, borderaxespad=0., ncol = 1, title = "Parameter") # 
ax.set(ylim = [-1.1,1.1], xlim = [-1.1,1.1])
# plt.legend(loc="best",
# frameon = False, shadow = False, borderaxespad=0., ncol = 2)
plt.tight_layout()
image_name = output_path / (f"{y_col_short}_PLSLoadingsScatter" + ".png")
plt.savefig(image_name, facecolor='w')
plt.show()
plt.rcParams.update({'font.size': 16})

#%%

#%% #################### DATE VIZ ####################

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

#%%


#%% functions
def pls_rfe(train,labels, cv_folds = 5, score_metric = 'neg_root_mean_squared_error'):
    """
    Returns: reg_est, reg_params, reg_rmse, labels_pred
    https://www.kaggle.com/miguelangelnieto/pca-and-regression
    """
    # results={}

    # def test_model(clf):
        
    #     cv = KFold(n_splits= cv_folds,shuffle=True,random_state=45)
    #     # r2 = make_scorer(r2_score)
    #     scores = (cross_val_score(clf, train, labels, cv=cv,scoring='neg_root_mean_squared_error'))
        
    #     return abs(scores.mean())
  
    # def rfe_grid_search(r_model, r_params):
        
    #     r = GridSearchCV(r_model, r_params,
    #         scoring = 'neg_root_mean_squared_error')
    #     r.fit(train, labels)
        
    #     return r#, r.best_estimator_, r.best_params_
    
    est = PLSRegression()
    selector = feature_selection.SequentialFeatureSelector(est, scoring = score_metric)#feature_selection.RFE(est, step = 1)
    pipe_params = [('feat_selection',selector),('clf', est)]
    pipe = pipeline.Pipeline(pipe_params)

    clf_parameters = {'feat_selection__n_features_to_select': list(range(1,8)),
    'clf__n_components': list(range(1,2))}

    sfs_grid = GridSearchCV(pipe, clf_parameters,
            scoring = score_metric, cv = cv_val) # 
    sfs_grid.fit(train, labels)

    best_pls = sfs_grid.best_estimator_['clf']
    best_featselect = sfs_grid.best_estimator_['feat_selection']
    best_params = sfs_grid.best_params_
    best_score = sfs_grid.best_score_ 

    print("Grid search complete.")
        
    return sfs_grid, best_featselect, best_pls, best_params, best_score

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

def pca_var_explained(X_array, number_components, 
x_lim = [0,10], save_fname = None, save_path = None):

    cov_mat = np.cov(X_array.T) # covariance matrix
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat) # decom
    tot = sum(eigen_vals)
    var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    print(cum_var_exp[:10]*100)
    print(f"{number_components} components explains {cum_var_exp[number_components-1]*100: .1f} % of total variation")

    # plotting
    fig, ax = plt.subplots(figsize = (8,8))
    x_vals = range(1, len(cum_var_exp)+1)

    ax.bar(x_vals, var_exp,
            alpha=1, align='center',
            label='Individual', color = 'grey')
    ax.step(x_vals, cum_var_exp, where = "mid", label='Cumulative', color = "black")
    
    plt.legend(loc='best')
    plt.grid(which = "both", axis = "y")
    ax.set(xlim = x_lim, ylim = [0,1], ylabel = 'Variance Explained', xlabel = 'Principal Components')
    plt.tight_layout()

    if save_path and save_fname:
        image_name = save_path / (save_fname.replace(" ", "")+".png")
        plt.savefig(image_name, facecolor='w')
    plt.show()

def sns_barplot(dataFrame, x_label, y_label, orient_var = "v", tick_rotate = 0,
title_var = None, save_fname = None, save_path = None):
    
    fig, ax = plt.subplots(figsize = (8,6))
    
    sns.barplot(x=x_label, y=y_label, data=dataFrame, ax=ax, color = "grey",
    orient = orient_var, edgecolor = "black") # palette = tab10_colors_list, edgecolor = "black",

    plt.grid(axis = "x")
    plt.xticks(rotation=tick_rotate)
        
    #ax.set_ylabel(y_label)
    ax.set_ylabel("")
    
    fig.suptitle(title_var)
        
    plt.tight_layout()
    
    if save_path and save_fname:
        image_name = save_path / (save_fname.replace(" ", "")+".png")
        plt.savefig(image_name, facecolor='w')
     
    plt.show()

def lets_try_pls(train,labels, cv_folds = 5):
    """
    Returns: reg_est, reg_params, reg_rmse, labels_pred
    https://www.kaggle.com/miguelangelnieto/pca-and-regression
    """
    results={}

    def test_model(clf):
        
        cv = KFold(n_splits= cv_folds,shuffle=True,random_state=45)
        # r2 = make_scorer(r2_score)
        scores = (cross_val_score(clf, train, labels, cv=cv,scoring='neg_root_mean_squared_error'))
        
        return abs(scores.mean())

    
    def grid_search(r_model, r_params):
        
        r = GridSearchCV(r_model, r_params,
            scoring = 'neg_root_mean_squared_error')
        r.fit(train, labels)
        
        return r.best_estimator_, r.best_params_

        # clf = PLSRegression(n_components = 2)
    
    clf_parameters = {'n_components':list(range(1,16))}
    clf_model = PLSRegression()
    reg_est, reg_params = grid_search(clf_model, clf_parameters)
    reg_rmse = test_model(reg_est)
    reg_coefs = reg_est.coef_
    labels_pred = cross_val_predict(reg_est, train, labels, cv = cv_folds)
        
    return reg_est, reg_params, reg_coefs, reg_rmse, labels_pred

def lets_try(train,labels, cv_folds = 5):
    """
    https://www.kaggle.com/miguelangelnieto/pca-and-regression
    """
    results={}

    def test_model(clf):
        
        cv = KFold(n_splits= cv_folds,shuffle=True,random_state=45)
        # r2 = make_scorer(r2_score)
        scores = (cross_val_score(clf, train, labels, cv=cv,scoring='neg_root_mean_squared_error'))
        return [abs(scores.mean())]

    
    def grid_search(r_model, r_params):
        
        r = GridSearchCV(r_model, r_params,
            scoring = 'neg_root_mean_squared_error')
        r.fit(train, labels)
        
        return r.best_estimator_, r.best_params_

    clf = linear_model.LinearRegression()
    results["Linear"]=test_model(clf)

    # clf = PLSRegression(n_components = 2)
    clf_parameters = {'n_components':list(range(1,16))}
    clf_model = PLSRegression()
    clf, clf_params = grid_search(clf_model, clf_parameters)
    results["PLS"] = test_model(clf)

    pca_comps_max = min([train.shape[0], train.shape[1]])
    clf = make_pipeline(PCA(n_components= pca_comps_max), linear_model.LinearRegression())
    results["PCR"]= test_model(clf)

    clf = linear_model.Ridge()
    results["Ridge"]=test_model(clf)
    
    clf = linear_model.BayesianRidge()
    results["Bayesian Ridge"]=test_model(clf)
    
    clf = linear_model.HuberRegressor()
    results["Hubber"]=test_model(clf)
    
    clf = linear_model.Lasso(alpha=1e-3)
    results["Lasso"]=test_model(clf)
    
    clf = BaggingRegressor()
    results["Bagging"]=test_model(clf)
    
    clf = RandomForestRegressor()
    results["RandomForest"]=test_model(clf)
    
    clf = AdaBoostRegressor()
    results["AdaBoost"]=test_model(clf)
    
    reg_search = {'kernel':['rbf'],
    'gamma':[0.0001, 0.001, 0.01,0.1,1,10],
    'gamma':["scale","auto"]
    }
    reg_est, reg_params = grid_search(svm.SVR(), reg_search)
    results["SVM RBF"]=test_model(reg_est)

    reg_search = {'kernel':['linear'],
    'gamma':[0.0001, 0.001, 0.01,0.1,1,10],
    'gamma':["scale","auto"]
    }
    reg_est, reg_params = grid_search(svm.SVR(), reg_search)
    results["SVM Linear"]=test_model(reg_est)
    
    # create results df
    results = pd.DataFrame.from_dict(results,orient='index')
    results.columns=["RMSE"]
    results.index.rename("Method", inplace = True)
    results.reset_index(drop = False, inplace = True)
    results.sort_values(by =["RMSE"], ascending=True, inplace = True)

    return results

def save_to_txt(save_fname, save_path, txt_string):
    txt_file_name = save_path / save_fname
    txt_file = open(txt_file_name, "w")
    txt_file.write(txt_string)
    txt_file.close()

    print(f"{save_fname} saved.")
    return

def time_var_loading_plot(x_vals, y_vals, c_vals,
x_label, c_label, clims = None, save_fname = None, save_path = None):
 
    fig, ax = plt.subplots(figsize = (8,6))
    
    im = ax.scatter(x_vals,
    y_vals, c = c_vals,
    cmap = "coolwarm_r", marker = 's', s = 200, alpha = 0.5,
    edgecolor = None, vmin = clims[0], vmax = clims[1])

    fig.colorbar(im, ax=ax, label = c_label)

    ax.set(ylabel ="", xlabel = x_label)
    ax.grid(which = "both", axis = "x")

    plt.tight_layout()
    
    if save_path and save_fname:
        image_name = save_path / (save_fname.replace(" ", "")+".png")
        plt.savefig(image_name, facecolor='w')
    plt.show()

def contribution_chart(data_variables, loadings, x_label, xlim = None,
save_path = None, save_fname = None):
    
    fig, ax = plt.subplots(figsize = (8,6))
    
    sns.barplot(x = loadings, y = data_variables,
        edgecolor = "black", orient = "h", color = "grey", ax = ax)
    ax.set_xlabel(x_label)
    ax.set_ylabel("")

    ax.set(xlabel = x_label, ylabel = "", xlim = xlim)

    # if xlim:
    #     plt.xlim(xlim)
        # dxlim = (max(xlim) - min(xlim))/8
        # plt.xticks(np.arange(min(xlim), max(xlim)+dxlim, dxlim))

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # ax.spines['left'].set_color('none')
    plt.grid(which = "both", axis = "x")
    plt.tight_layout()
    
    if save_path and save_fname:
        image_name = save_path / (save_fname + ".png")
        plt.savefig(image_name, facecolor='w')
     
    plt.show()

def parity_res_plots(y_val_train, y_val_train_pred, y_val_train_res,
    response_var_label,
    y_val_test = None, y_val_test_pred = None, y_val_test_res = None,
    y_val_train_label = None, y_val_test_label = None,
    save_fname = None, save_path = None,
    marker_s_var = 90, zero_axis_lim = True):

    fig, axs = plt.subplots(1, 2, figsize = (12,6))

    if y_val_test is not None:
        all_ys = list(y_val_train) + list(y_val_train_pred)+\
        list(y_val_test_pred) + list(y_val_test)
        
    else:
        all_ys = list(y_val_train) + list(y_val_train_pred)

    axis_lims = [min(all_ys)*0.9, max(all_ys)*1.1]

    def parity_plot(ax):
        # plt.style.use('default')

        ax.scatter(y_val_train, y_val_train_pred,
            linewidth = 1.5, marker = "o",color = tab10_colors_list[0], facecolor = "none",
            s = marker_s_var, label = y_val_train_label)

        ax.plot(axis_lims, axis_lims, linestyle = "--", color = "grey")

        if y_val_test is not None:

            ax.scatter(y_val_test, y_val_test_pred,
            linewidth = 1.5, marker = "^", color = tab10_colors_list[1], facecolor = "none",
            s = marker_s_var, label = y_val_test_label)

            plt.legend(loc = "best")

        if zero_axis_lim == False:
            ax.set_ylim(axis_lims)
            ax.set_xlim(axis_lims)
        else:
            ax.set_ylim(0, axis_lims[1])
            ax.set_xlim(0, axis_lims[1])

        ax.set(xlabel = f"{response_var_label}", ylabel= f"Predicted {response_var_label}")

    
    def plot_res(ax):

        ax.scatter(y_val_train_pred, y_val_train_res,
            linewidth = 1.5, marker = "o",color = tab10_colors_list[0], facecolor = "none",
            s = marker_s_var, label = y_val_train_label)

        ax.plot(axis_lims, [0,0], linestyle = "-", linewidth = 1, color = "red")
        ax.plot(axis_lims, [2,2], linestyle = "--", linewidth = 1, color = "red")
        ax.plot(axis_lims, [-2,-2], linestyle = "--", linewidth = 1, color = "red")

        if y_val_test_res is not None:

            ax.scatter(y_val_test_pred, y_val_test_res,
            linewidth = 1.5, marker = "^",color = tab10_colors_list[1], facecolor = "none",
            s = marker_s_var, label = y_val_test_label)

            plt.legend(loc = "best")

        # ax.spines['bottom'].set_position('zero')
        #ax.spines['right'].set_color('none')
        #ax.spines['top'].set_color('none')
        # ax.grid(which = "both", axis = "both")
        ax.set(xlabel = f"Predicted {response_var_label}",
            ylabel = f"Standardised Residuals", xlim = axis_lims)

        # ax.grid(which = "both", axis = "both")
        #plt.tight_layout()

        return (ax)

    parity_plot(axs[0])
    plot_res(axs[1])
    fig.tight_layout()

    if save_path and save_fname:
        image_name = save_path / (save_fname.replace(" ", "")+".png")
        plt.savefig(image_name, facecolor='w')

    plt.show()

def standardised_res(obs_y, pred_y):
        res_y = obs_y - pred_y
        std_res_y = res_y / (res_y).std()
        return std_res_y

def subplots_scatter(dataFrame, cols_X, col_y, n_cols = 3,
title_var = None, save_fname = None, save_path = None):
        
    n_rows = math.ceil(len(cols_X)/n_cols)
    height_figure = 3*n_rows
    fig, ax = plt.subplots(n_rows,n_cols, figsize=(10, height_figure),
                    sharey = True, sharex = False)

    for i, param in enumerate(cols_X):

        ax.ravel()[i].scatter(dataFrame[param], dataFrame[col_y],
            color = "black")
        #ax.ravel()[i].xaxis.set_major_formatter(myFmt)
        #ax.ravel()[i].tick_params(axis='x', rotation=45)
        # ax.ravel()[i].legend(loc = "upper left")
        # ax.ravel()[i].grid(axis = "y")
        ax.ravel()[i].set(xlabel = param)
        # ax.ravel()[i].set_xlim(
        # pd.to_datetime("2021-01-19 14:00:00"),
        # pd.to_datetime("2021-09-01 16:45:00"))

        if i % n_cols == 0:
            ax.ravel()[i].set(ylabel = col_y)
    # plots to remove
    plot_remove = n_rows*n_cols - len(cols_X)
    for i in range(1, plot_remove+1):
        ax.ravel()[i*-1].set_visible(False)
    
    plt.suptitle(title_var)
    # plt.xticks(rotation = 45)
    plt.tight_layout()

    if save_path:
        if save_fname == None:
            save_fname = title_var.replace("/"," ")
        
        image_name = save_path / (save_fname.replace(" ", "")+".png")
        plt.savefig(image_name, facecolor='w')
    
    plt.show()

def F_squarred(x, A):
    y = x**(A)
    return y

def diff_F_squrred(x, A):
    y = (A)*x**(A-1)
    return y

def create_dir(dir_name, parent_dir):

    new_dir_path = Path(os.path.join(parent_dir, dir_name))
    try:
        os.mkdir(new_dir_path)
    except OSError:
        print(f"{dir_name} exists.")
    
    return new_dir_path

def Scatter3D_cat(dataFrame, x_label, y_label, z_label, c_label, 
    save_fname = None, save_path = None):
 
    fig = plt.figure(figsize = (7,7)) # dpi = 600
    ax = fig.add_subplot(projection = "3d")

    m_list = ["o", "D"]
    for i, c_val in enumerate(dataFrame[c_label].unique()):
        dataFrame_plot = dataFrame.loc[dataFrame[c_label] == c_val]

        ax.scatter(dataFrame_plot[x_label],dataFrame_plot[y_label],dataFrame_plot[z_label],
            color = tab10_colors_list[i], marker = marker_list[i],
            s = 140, alpha = 0.4, edgecolor = tab10_colors_list[i], label = c_val)

    ax.view_init(20, 30)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_zlabel(z_label)
    ax.tick_params(labelsize=12)
    plt.legend(title = c_label, loc = "upper left",
        bbox_to_anchor = (1.,0.8), frameon = False)

    # fig.tight_layout()
    
    if save_path and save_fname:
        image_name = save_path / (save_fname.replace(" ", "")+".png")
        plt.savefig(image_name, facecolor='w')
    
    plt.show()

def scatter_line_plot(x_vals, y_vals, x_label, y_label, y_std = None,
    y_lims = [0,15], x_lims = None, line_width = 1,
    y_scale = 'linear', save_fname = None, save_path = None):
    
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot(x_vals, y_vals, linewidth = line_width, color = "black", marker = "o")


    if y_std is not None:
        try:
            ax.fill_between(x_vals, (y_vals + y_std), (y_vals - y_std),
                color = "grey", alpha = 0.4)
        except:
            pass

    ax.set(xlabel = x_label, ylabel = y_label)
        #ylim = y_lims, xlim = x_lims, yscale = y_scale)

    ax.grid(which = "both", axis = "both")
    fig.tight_layout()

    if save_path and save_fname:
        image_name = save_path / (save_fname.replace(" ", "")+".png")
        plt.savefig(image_name, facecolor='w')
     
    plt.show()

def pcr(train, labels, cv_folds = 5, score_metric = 'neg_root_mean_squared_error'):

    # est = 
    # pca = 
    #pipe_params = [('pca',pca),('est', est)]
    pipe = pipeline.Pipeline([('pca',PCA()),('est', LinearRegression())])

    grid_params = {"pca__n_components": list(range(1,10))}

    pcr_grid = GridSearchCV(pipe, grid_params,
            scoring = score_metric, cv = cv_val) # 
    pcr_grid.fit(train, labels)

    #best_linreg = pcr_grid.best_estimator_['est']
    best_pca = pcr_grid.best_estimator_['pca']
    best_params = pcr_grid.best_params_
    best_score = pcr_grid.best_score_ 
    print("Grid search complete.")
    # best_linreg,
        
    return pcr_grid, best_pca, best_params, best_score


def pca_biplot(x_vals, y_vals, x_label, y_label, c_val = None, c_label = None, cmap_var = "viridis_r",
save_fname = None, save_path = None):
    
    fig, ax = plt.subplots(figsize = (10,6))

    if c_label:
        
        edgecolor_var = "black"
        im = ax.scatter(x_vals, y_vals, c = c_val,
            cmap = cmap_var, marker = 'o', s = 100, alpha = 0.7,linewidth = 1.5,
            edgecolor = edgecolor_var, label = c_label)
            
        fig.colorbar(im, ax=ax, label = f"{c_label}")
    else:
        edgecolor_var = "black"#None
        im = ax.scatter(x_vals, y_vals, marker = 'o', s = 100,
            facecolor = "none", linewidth = 1.5,
            edgecolor = edgecolor_var)

    ax.axhline(y=0, color = "black", linewidth = 1.5)
    ax.axvline(x=0, color = "black", linewidth = 1.5)
    ax.set(ylabel = y_label, xlabel = x_label)

    plt.tight_layout()
    plt.grid(which = "both", axis = "both")
    
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

def basic_scatter(dataFrame, x_string, y_string, z_string = None,
                   marker_size = 50, y_limits = None, x_limits = None,
                   save_fname = None, save_path = None):

    fig, ax = plt.subplots(figsize = (7,5))
    
    if z_string:
        z_list = dataFrame[z_string].unique()
        for i, z in enumerate(z_list):
            x_vals = dataFrame[x_string].loc[dataFrame[z_string] == z]
            y_vals = dataFrame[y_string].loc[dataFrame[z_string] == z]

            ax.scatter(x_vals ,y_vals,
                    color = tab10_colors_list[i], marker = marker_list[i], facecolor = "none",
                    label = str(z), s = marker_size)
            
        ax.legend(title = z_string, loc = "best",
                  framealpha = 1, shadow = False)
    
    else:
        ax.scatter(dataFrame[x_string] ,dataFrame[y_string],
                color = tab10_colors_list[0], marker = marker_list[0], facecolor = "none",
                s = marker_size)
        
    # ax.grid(which = "both", axis = "y")
    ax.set(ylim = y_limits, xlim = x_limits,
           ylabel = y_string, xlabel = x_string)
    
    plt.tight_layout()
    
    if save_path and save_fname:
        image_name = save_path / (save_fname.replace(" ", "")+".png")
        plt.savefig(image_name, facecolor='w')
    
    plt.show()
