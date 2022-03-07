"""
Imports unfolded fermentation dataset
Applies PCA to explore sources of variance
"""
#%% imports
from JMTCi_Functions import DoPCA, DoPickle, create_dir, biplot
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

cb_colors_list = sns.color_palette("muted")+sns.color_palette("muted")
plt.style.use('default')
plt.rcParams.update({'font.size': 16})

#%% functions
def corr_heatmap(dataFrame_corr,
    figsize_var = (8,4), vrange = [-1, 1], cmap_var = "coolwarm_r", save_fname = None, save_path = None):

    fig, ax = plt.subplots(figsize = figsize_var)
    sns.heatmap(dataFrame_corr, linewidths=.5,
        vmin = vrange[0], vmax = vrange[1], annot = True,
        cmap = cmap_var, cbar = False,fmt='.2f',ax = ax)

    plt.tight_layout()

    if save_path and save_fname:
        image_name = save_path / (save_fname.replace(" ","") + ".png")
        
        plt.savefig(image_name, facecolor='w')
     
    plt.show()

def find_time_in_string(row_string, time_marker = ", t="):

    if time_marker in row_string:
        return (row_string.split(time_marker)[0], float(row_string.split(time_marker)[1]))
    else:
        return (row_string, np.nan)

#%% input variable
source_path = Path("C:/Users/JacksC01/OneDrive - Johnson Matthey/Documents/DigitalModelling/JMTCi4Fermentation/data")
output_path = Path("C:/Users/JacksC01/OneDrive - Johnson Matthey/Documents/DigitalModelling/JMTCi4Fermentation/figures")

source_fname = "batchwise_fermentation.csv"
col_fname = "AllColsDict.pkl"

pca_components = 4

#%% import of production summary and clean
df = pd.read_csv(source_path / source_fname)
df.set_index("USP", drop = True, inplace = True)

all_cols_dict = DoPickle(col_fname).pickle_load()

y_cols = all_cols_dict['y_cols']
X_cols_tvar = all_cols_dict['TimeVarParams']
X_cols_tinvar = all_cols_dict['TimeInvarParams']
X_cols = X_cols_tinvar + X_cols_tvar
df.sort_values(by = ["Scale (l)"]+ y_cols, ascending = True, inplace = True)

#%% PCA data prep
pca_path = create_dir("PCA", output_path)

X_cols_pca = list(set(X_cols))# - set(X_cols_tvar))
X = df[X_cols_pca].to_numpy()
scaler_X = StandardScaler()
scaler_X.fit(X) # get mean and std dev for scaling
X_sc = scaler_X.transform(X) # scaled

usp_index = df.index.to_list()

y = df[y_cols].to_numpy()
scaler_y = StandardScaler()
scaler_y.fit(y) # get mean and std dev for scaling
y_sc = scaler_y.transform(y) # scaled

print("Data ready.")

#%%
pca_object = DoPCA(X_sc)
pca_object.plot_var_explained(x_lim = [0,8], save_path = pca_path, save_fname = "BWUPCAExplainedVariance")
loadings_pca, scores_pca, eigen_vals_pca, pca_cols, pca_exp_cols = pca_object.calibrate_pca(pca_components)

pca_df = pd.DataFrame(data = scores_pca, columns = pca_cols, index = usp_index)
pca_df = pd.merge(left = pca_df, right = df[y_cols+["Scale (l)"]], left_index = True,
    right_index = True)
pca_df["USP"] = pca_df.index
pca_df.reset_index(drop = True, inplace = True)

# pca_df_melt = pd.melt(pca_df, id_vars = ["USP","Scale (l)"]+y_cols,
#     var_name = "Component", value_name = "Value")

df_loadings = pd.DataFrame(loadings_pca, columns= pca_cols, index = X_cols_pca)
df_loadings["Feature"] = df_loadings.index
df_loadings["Parameter"] = df_loadings["Feature"].apply(lambda x: x.rsplit(", t", 2)[0])

#%%
df_loadings["Time Batch (h)"] = df_loadings["Feature"].apply(lambda x: find_time_in_string(x, ", tb=")[1])
df_loadings["Time Feed (h)"] = df_loadings["Feature"].apply(lambda x: find_time_in_string(x, ", tf=")[1])
df_loadings["Time Induction (h)"] = df_loadings["Feature"].apply(lambda x: find_time_in_string(x, ", ti=")[1])
df_loadings_melt = pd.melt(df_loadings, id_vars=["Parameter", "Time Batch (h)","Time Feed (h)","Time Induction (h)"], value_vars = pca_cols,
var_name="PC", value_name='Loadings', col_level=None, ignore_index=True)

max_loading = df_loadings_melt['Loadings'].abs().max()

df_loadings_melt.sort_values(by = ["Parameter"], inplace = True, ascending = False)
# df_loadings_melt['Loadings'] = df_loadings_melt['Loadings'].abs()
loadings_range = (max_loading*-1, max_loading)
# loadings_range = (0, max_loading)

#%% plot time variant loadings
time_scales = ["Time Batch (h)", "Time Feed (h)", "Time Induction (h)"]

plt.rcParams.update({'font.size': 14})
fig, ax = plt.subplots(pca_components, len(time_scales), figsize = (12, 13), sharey = True,
                       gridspec_kw={
                           'width_ratios': [9/9,16/9,24/9]})

for i, pc in enumerate(pca_cols):

    for j, time_scale in enumerate(time_scales):

        sns.scatterplot(ax = ax[i,j],
            data = df_loadings_melt.loc[(~df_loadings_melt[time_scale].isna()) & (df_loadings_melt["PC"] == pc)],
            x = time_scale, y = "Parameter", hue = 'Loadings', 
            palette = "vlag_r", hue_norm = loadings_range, marker = "s", s = 150, alpha = 1, edgecolor = 'none') # "coolwarm_r"

        ax[i,j].legend([],[], frameon=False)

        ax[i,j].set(ylabel = "", xlabel = "")
        ax[i,j].grid(which = "both", axis = "x")
    
    ax[i,0].set(ylabel = pca_exp_cols[i])

for j, time_var in enumerate([r"$t_{b}$ (h)",r"$t_{f}$ (h)",r"$t_{i}$ (h)"]):
    ax[-1,j].set(xlabel = time_var)

fig.tight_layout()

image_name = pca_path / ("timevar_loadings".replace(" ", "")+".png")
plt.savefig(image_name, facecolor='w')
plt.show()

#%% plot time invariant loadings 
df_timeinvar = df_loadings_melt[df_loadings_melt["Parameter"].isin(X_cols_tinvar)]#.drop(X_cols_tvar, axis = 0)#df_loadings.loc[df_loadings.index.isin(X_cols_tinvar)]
loading_thresh = max_loading/3
df_timeinvar["Significant"] = np.where(df_timeinvar["Loadings"].abs() > loading_thresh,
    True, False)

g = sns.catplot(data = df_timeinvar.sort_values(by = ["PC", "Parameter"], ascending = True), kind = "bar",
    x = "Loadings", y = "Parameter", col = "PC", hue = "Loadings",#"Significant",
    col_wrap = 2, palette = "vlag_r",#sns.color_palette(["white","grey"]),
    dodge = False, edgecolor = "black",
    orient = "h", height = 4, aspect = 1.5, ci = None, legend = False)

for ax in g.axes.flat:
    ax.grid(which = "major", axis = "y")

g.set(ylabel = "", xlim = loadings_range)
g.set_titles("{col_name}")
#g.despine(left=True)
image_name = pca_path / ("timeinvar_loadings".replace(" ", "")+".png")
g.savefig(image_name, facecolor='w')
plt.show()

#%% pc heat map
corr = pca_df[pca_cols + y_cols].corr()
corr = corr[pca_cols].loc[corr.index.isin(y_cols)]
corr_heatmap(corr, save_fname = "PCHeatMap", save_path = pca_path)

#%% PC biplots
biplot(pca_df["PC1"], pca_df["PC4"], pca_exp_cols[0], pca_exp_cols[3],
c_val = y_sc[:,y_cols.index('Biomass Density (g/l)')], c_label = "Biomass Density",
cmap_var = "viridis_r",
save_fname = f"BiomassDensity_PC1PC4", save_path = pca_path)

biplot(pca_df["PC1"], pca_df["PC4"], pca_exp_cols[0], pca_exp_cols[3],
c_val = y_sc[:,y_cols.index('Harvest WCW (g/l)')], c_label = 'Harvest WCW',
cmap_var = "viridis_r",
save_fname = f"HarvestWCW_PC1PC4", save_path = pca_path)

biplot(pca_df["PC1"], pca_df["PC3"], pca_exp_cols[0], pca_exp_cols[2],
c_val = y_sc[:,y_cols.index('Volumetric Activity (U/ml)')], c_label = "Volumetric Activity",
cmap_var = "viridis_r",
save_fname = f"VolumetricActivity_PC1PC3", save_path = pca_path)

# for y_col in y_cols:
#     biplot(pca_df["PC1"], pca_df["PC2"], pca_exp_cols[0], pca_exp_cols[1],
#         c_val = y_sc[:,y_cols.index(y_col)], c_label = y_col.split(" (")[0],
#         cmap_var = "viridis_r",
#         save_fname = None, save_path = pca_path)

#     biplot(pca_df["PC2"], pca_df["PC3"], pca_exp_cols[1], pca_exp_cols[2],
#         c_val = y_sc[:,y_cols.index(y_col)], c_label = y_col.split(" (")[0],
#         cmap_var = "viridis_r",
#         save_fname = None, save_path = pca_path)

#%% PC 3D scatter plots
plt.rcParams.update({'font.size': 12})

fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(1, 2, 2, projection='3d')

ax1.scatter(pca_df["PC1"], pca_df["PC4"], y_sc[:,y_cols.index('Volumetric Activity (U/ml)')],
    label = "Volumetric Activity",
    color = cb_colors_list[1], edgecolor = cb_colors_list[1], marker = 'o', s = 150, alpha = 0.7, linewidth = 1.5)

ax2 = fig.add_subplot(1, 2, 1, projection='3d', sharez = ax1)
ax2.scatter(pca_df["PC1"], pca_df["PC4"], y_sc[:,y_cols.index('Harvest WCW (g/l)')],
    label = 'Harvest WCW (g/l)',
    color = cb_colors_list[0], edgecolor = cb_colors_list[0], marker = 'o', s = 150, alpha = 0.7, linewidth = 1.5)

handles, labels = [], []

for ax, y_col in zip([ax1, ax2], ["Volumetric Activity", 'Harvest WCW']):
    
    ax.view_init(20, 60)
    #, zlim = [-2,2], xlim = [-3.5,3.5], ylim = [-3.,3.]
    ax.set(xlabel = "PC1", ylabel = "PC4", zlabel = y_col,
        xlim = [-15,30], ylim = [-15,30])

    h, l = ax.get_legend_handles_labels()
    handles.append(h[0])
    labels.append(l[0])

image_name = pca_path / ("PC1PC4_3DScatter".replace(" ", "")+".png")
plt.savefig(image_name, facecolor='w')
plt.show()

#%% pc distribution
# fig, ax = plt.subplots(1,2, figsize = (10, 5), sharey = True)
# sns.histplot(ax = ax[0], data = pca_df, x = "PC1", hue = "Scale (l)",
#     palette = sns.color_palette("colorblind")[2:4], bins = 20)
# sns.histplot(ax = ax[1], data = pca_df, x = "PC2", hue = "Scale (l)",
#     palette = sns.color_palette("colorblind")[2:4], bins = 20)
# plt.show()

# g = sns.catplot(data = df_timeinvar.sort_values(by = ["PC", "Parameter"], ascending = True), kind = "bar",
#     x = "Loadings", y = "Parameter", col = "PC", hue = "Significant",
#     col_wrap = 2, palette = sns.color_palette(["white","grey"]),
#     dodge = False, edgecolor = "black",
#     orient = "h", height = 4, aspect = 2.5, ci = None, legend = False)
# for ax in g.axes.flat:
#     ax.grid(which = "major", axis = "y")
# g.set(ylabel = "", xlim = loadings_range)
# g.set_titles("{col_name}")
# image_name = pca_path / ("timeinvar_loadings".replace(" ", "")+".png")
# g.savefig(image_name, facecolor='w')
# plt.show()

#%%
print("Done.")

#%%