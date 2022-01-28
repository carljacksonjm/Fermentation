"""
base pump , contin / discrete 
some media better for growth than others?
optimal pcas for each stage
pc values for each stage
"""
#%% imports
from matplotlib.markers import MarkerStyle
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns

import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import warnings
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
plt.style.use('default')
plt.rcParams.update({'font.size': 16})

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

def scatter_2D(x_vals, y_vals, x_label, y_label, xlim = None, c_val = None, c_label = None,
title_var = None, save_path = None):
    
    if c_label:
        fig, ax = plt.subplots(figsize = (8,6))
        edgecolor_var = "black"

    else:
        fig, ax = plt.subplots(figsize = (6,6))
        edgecolor_var = None

    im = ax.scatter(x_vals, y_vals, c = c_val,
    cmap = "cividis", marker = 'o', s = 100, alpha = 0.7,
    edgecolor = edgecolor_var, label = c_label)

    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)

    if c_label:
        fig.colorbar(im, ax=ax, label = f"{c_label}")
      
    if title_var:
        plt.title(title_var)

    plt.xlim(xlim) 
    plt.tight_layout()

    plt.grid(which = "both", axis = "both")
    
    if save_path:
        image_name = save_path / (title_var.replace(" ", "")+".png")
        plt.savefig(image_name, facecolor='w')
        
    plt.show()

def Scatter3D(x_vals, y_vals, z_vals,
x_label, y_label, z_label,
c_val = None, c_label = None, title_var = None, save_fname = None, save_path = None):
 
    fig = plt.figure(figsize = (10,8)) # dpi = 600
    ax = fig.add_subplot(projection = "3d")
    
    if c_label:

        im = ax.scatter(x_vals,
                    y_vals,
                    z_vals, c = c_val, 
                cmap = "cividis", marker = 'o',
                s = 100, alpha = 1, edgecolor = "black", label = c_label)
        
        fig.colorbar(im, ax=ax, label = f"{c_label}")
    else:
        im = ax.scatter(x_vals,
                    y_vals,
                    z_vals,  
                marker = 'o',
                s = 100, alpha = 1, edgecolor = "black", color = jm_colors_list[0])
    ax.view_init(20, 30)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_zlabel(z_label)  
      
    if title_var:
        plt.title(title_var)
        
    plt.tight_layout()
    
    if save_path:
        if save_fname == None:
            save_fname = title_var
        else:
            pass

        image_name = save_path / (save_fname.replace(" ", "")+".png")
        plt.savefig(image_name, facecolor='w')
    
    plt.show()

def sns_lineplot(dataFrame, x_label, y_label, color_label, style_label = None, legend_var = False, title_var = None, save_path = None):
    
    fig, ax = plt.subplots(figsize = (10,6))
    
    sns.lineplot(x=x_label, y=y_label, hue = color_label, style = style_label, data=dataFrame, ax=ax,
    legend = True)

    plt.grid(axis = "both")

    if legend_var == True:
            plt.legend(title = color_label, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    else:
        plt.legend([],[], frameon=False)
        
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    
    if title_var:
        fig.suptitle(title_var)
        
    plt.tight_layout()
    
    if save_path:
        image_name = save_path / (title_var.replace(" ", "")+".png")
        plt.savefig(image_name, facecolor='w')
     
    plt.show()

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
    plt.ylabel('Explained Variance')
    plt.xlabel('Principal Components')
    plt.legend(loc='best')
    plt.grid(which = "both", axis = "y")

    # if title_var:
    #     plt.title(title_var)

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


#%% input variables
#%% input variable
source_path = Path("C:/Users/JacksC01/OneDrive - Johnson Matthey/Documents/DigitalModelling/JMTCi4Fermentation/data")
output_path = Path("C:/Users/JacksC01/OneDrive - Johnson Matthey/Documents/DigitalModelling/JMTCi4Fermentation/figures")

source_fname = "obswise_fermentation.csv"

#%% import of production summary and clean
df = pd.read_csv(source_path / source_fname)
df.drop(columns = ["Unnamed: 0"], inplace = True)
df.set_index(["USP","Time (H)"], drop = True, inplace = True)
df.sort_values(by = ["USP","Time (H)"], ascending= True, inplace = True)

features = ['Volumetric Activity (U/mL)',
 'Harvest x_bar WCW (g/L)',
 'Biomass Activity (U/g)',
 'Total Biomass (g)']

X_cols = list((set(df.columns) - set(features)) - {"USP", "Time (H)"})

X_cols_ps = ['Scale',
 'Induction Temp (°C)',
 'Induction pH',
 'Feed Seq Red at Induct (%)',
 'Feed Media_F04',
 'Glycerol ID_MGS-0467',
 'Glycerol ID_WGS-0020',
 'Production Media_gMSMK',
 'Production Media_gMSMK mod1',
 'Production Media_gMSMK mod2',
 'Antibiotic_Kan',
 'Antibiotic_None',
 'Feed Seq Step_Ramp',
 'Feed Seq Step_Steps']

X_cols_trends = list((set(X_cols) - set(X_cols_ps)))

#%% pca prep
X = df[X_cols].to_numpy()
scaler_X = StandardScaler()
scaler_X.fit(X) # get mean and std dev for scaling
X_scaled = scaler_X.transform(X) # scale
# X_scaled = X

y = df[features].to_numpy()

# usp_index = df.USP.to_list()
# time_index = df["Time (H)"].to_list()
df_index = df.index

pca_components = 5
print("Data ready.")

#%% pca variance explained
pca_var_explained(X_scaled, pca_components, x_lim = [0,10],
title_var = "OWU PCA Explained Variance", save_path = output_path)

#%% pca
loadings_pca, scores_pca, eigen_vals_pca = calibrate_pca(X_scaled, pca_components)
print(f"Loadings {loadings_pca.shape}\nScores {scores_pca.shape}\nEigen vals{eigen_vals_pca.shape}")
# loadings (variables, pca components)
# scores (usps, pca components) - can add features
# eigen (pca components)

pca_cols = ["PC{0}".format(i) for i in range(1,scores_pca.shape[1]+1)]
pca_df = pd.DataFrame(data = scores_pca, columns = pca_cols, index = df_index)

pca_df = pd.merge(left = pca_df, right = df, how = "left", left_index = True, right_index = True)
pca_df.reset_index(drop = False, inplace = True)

#%%
loadings_df = pd.DataFrame(loadings_pca, columns= pca_cols, index = X_cols)
loadings_df["PARAM"] = loadings_df.index

#%%
# for f in features:
#     print("-----------------------\n",f)

#     for col in pca_cols:
        
#         xlims = [pca_df[col].min()*0.9, pca_df[col].max()*1.1]

#         scatter_2D(pca_df[col], pca_df[f], col, f,
#             c_val = None, c_label = None, xlim = xlims,
#             title_var = None, save_path = None)

# scatter_2D(pca_df["PC3"], pca_df["Total Biomass (g)"], "PC3", "Total Biomass (g)",
#     c_val = pca_df["Total Biomass (g)"], c_label = "Total Biomass (g)",
#     title_var = "Total Biomass - PC3", save_path = output_path)

# scatter_2D(pca_df["PC1"], pca_df["Biomass Activity (U/g)"], "PC1", "Biomass Activity (U/g)",
#     c_val = pca_df["Biomass Activity (U/g)"], c_label = "Biomass Activity (U/g)",
#     title_var = "Mass Activity - PC1", save_path = output_path)

# scatter_2D(pca_df["PC1"], pca_df["Volumetric Activity (U/mL)"], "PC1", "Volumetric Activity (U/mL)",
#     c_val = pca_df["Volumetric Activity (U/mL)"], c_label = "Volumetric Activity (U/mL)",
#     title_var = "Volumetric Activity - PC1", save_path = output_path)


#%% plot time variant loadings 
max_loading = (loadings_df[pca_cols].abs().max().max()*1.)
loadings_range = [max_loading*-1, max_loading]

for pc in pca_cols:

    contribution_chart(loadings_df["PARAM"], loadings_df[pc], "Loadings", xlim = loadings_range,
    title_var = f"OWU Component Loadings - {pc}", save_path = output_path, save_fname = f"OWU_Loadings_{pc}")

#%% plot PC profiles as func of time
def sns_lineplot(dataFrame, x_label, y_label, color_label, palette_var = "crest", legend_var = False, 
legend_title_var = None, title_var = None, save_path = None, save_fname = None):

    # plt.style.use('default')
    
    fig, ax = plt.subplots(figsize = (14,8))
    
    # "rocket_r" sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
    sns.lineplot(x=x_label, y=y_label, hue = color_label, data=dataFrame, ax=ax,
    legend = True, palette = palette_var, dashes = True)

    if legend_title_var == None:
        legend_title_var = color_label 

    if legend_var == True:
            #plt.legend(title = color_label, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.legend(title = legend_title_var, loc = "upper left")
    else:
        plt.legend([],[], frameon=False)

    # xlims = [dataFrame[x_label].min(), dataFrame[x_label].max()] 
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    # ax.set_xlim(xlims[0], xlims[1])
    # ax.set_xlim(-30,0)

    if title_var:
        fig.suptitle(title_var)
        
    plt.tight_layout()
    plt.grid(which = "both", axis = "x")
    
    if save_path:
        print("saving...")
        if save_fname:
            image_name = save_path / (save_fname + ".png")
        else:
            image_name = save_path / (title_var.replace(" ", "")+".png")
        plt.savefig(image_name, facecolor='w')
        
    plt.show()

for f in features:
    sns_lineplot(pca_df, "Time (H)", "PC1", f, palette_var = "crest", legend_var = True, 
        legend_title_var = None, title_var = None, save_path = None, save_fname = None)

    sns_lineplot(pca_df, "Time (H)", "PC2", f, palette_var = "crest", legend_var = True, 
        legend_title_var = None, title_var = None, save_path = None, save_fname = None)

    print("---------------------")

# %%
f = features[2]
pca_df.sort_values(by = [f,"USP","Time (H)"],
inplace = True, ascending = False)

for i, usp in enumerate(pca_df.USP.unique()):

    df_plot = pca_df.loc[pca_df.USP == usp]
    f_val = df_plot[f].mean()
    
    scatter_2D(df_plot["PC1"], df_plot["PC2"], "PC1", "PC2",
        c_val = df_plot["Time (H)"], c_label = "Batch Time (h)",
        title_var = f"{usp} USP - {f} {f_val: .1f}", save_path = None)

# %%
