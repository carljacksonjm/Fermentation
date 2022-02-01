#%% imports
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import linregress
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

#%%
# jm_colors_dict = {
#     'JM Blue':'#1e22aa',
#     'JM Purple':'#e50075',
#     'JM Cyan':'#00abe9','JM Green':'#9dd3cb',
#     'JM Magenta':'#e3e3e3',
#     'JM Light Grey':'#575756',
#     'JM Dark Grey':'#6e358b'
# }
# jm_colors_list = list(jm_colors_dict.values())
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

def remove_date(row_str):
    if " " in row_str:
        row_str = str(row_str).split(" ")[-1]
    return str(row_str)

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

def clean_nasty_floats(nasty_series):
    good_series = nasty_series.replace("\t","", regex = True)
    good_series = good_series.replace(" ","", regex = True)

    return pd.to_numeric(good_series)

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

# %%


# def calibrate_pca(X_vals, n_components):

#     pca = PCA(n_components = n_components)
#     pca_scores = pca.fit_transform(X_vals) # PCA scores
#     pca_loadings = pca.components_.T # eigenvectors  aka loadings
#     eigen_values = pca.explained_variance_ # eigenvalues
#     print(f"X values: {X_vals.shape}")
#     print(f"Eigenvectors (loadings): {pca_loadings.shape}\nEigenvals: {eigen_values.shape}\nScores: {pca_scores.shape}")
    
#     return pca_loadings, pca_scores, eigen_values

# def check_day_month_order(date_series, identifier, list_month_first):

#     date_series_clean = date_series.apply(lambda x: str(x).replace("-","/"))

#     if identifier in list_month_first:
#         print(f"Month first {identifier}")

#         return pd.to_datetime(date_series_clean, format='%Y/%d/%m %H:%M:%S')
#     else:
#         #print("day first apparently: ", date_series_clean[:3])
#         return pd.to_datetime(date_series_clean, dayfirst= True) # , format='%d/%m/%Y %H:%M:%S'

# def strings_to_floats(string_series):
#     """converts to string
#     replaces ' and ,
#     converts to float    
#     """

#     string_series = str(string_series)
#     clean_string = string_series.replace("'","").replace(",","")
#     return float(clean_string)





# def sns_barplot(dataFrame, x_label, y_label, orient_var = "v",
# title_var = None, save_path = None):
    
#     fig, ax = plt.subplots(figsize = (6,6))
    
#     sns.barplot(x=x_label, y=y_label, data=dataFrame, ax=ax, palette = tab10_colors_list, edgecolor = "black", orient = orient_var)

#     # plt.grid(axis = "y")
        
#     ax.set_ylabel(y_label)
#     ax.set_xlabel(x_label)
    
#     if title_var:
#         fig.suptitle(title_var)
        
#     plt.tight_layout()
    
#     if save_path:
#         image_name = save_path / (title_var.replace(" ", "")+".png")
#         plt.savefig(image_name, facecolor='w')
     
#     plt.show()

# def contribution_chart(data_variables, loadings, x_label, xlim = None, title_var = None,
# save_path = None, save_fname = None):
    
#     fig, ax = plt.subplots(figsize = (12,6))
    
#     sns.barplot(x = loadings, y = data_variables, edgecolor = "black", orient = "h", color = tab10_colors_list[0], ax = ax)
#     ax.set_xlabel(x_label)
#     ax.set_ylabel("")

#     if xlim:
#         plt.xlim(xlim)
#         # dxlim = (max(xlim) - min(xlim))/8
#         # plt.xticks(np.arange(min(xlim), max(xlim)+dxlim, dxlim))

#     plt.title(title_var)
#     ax.spines['right'].set_color('none')
#     ax.spines['top'].set_color('none')
#     ax.spines['left'].set_color('none')
#     plt.grid(which = "both", axis = "x")
#     plt.tight_layout()
    
#     if save_path:
#         print("saving...")
#         if save_fname:
#             image_name = save_path / (save_fname + ".png")
#         else:
#             image_name = save_path / (title_var.replace(" ", "")+".png")
        
#         plt.savefig(image_name, facecolor='w')
     
#     plt.show()

# def pca_var_explained(X_array, number_components, x_lim = [0,10],title_var = None, save_path = None):

#     cov_mat = np.cov(X_array.T) # covariance matrix
#     eigen_vals, eigen_vecs = np.linalg.eig(cov_mat) # decom
#     tot = sum(eigen_vals)
#     var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
#     cum_var_exp = np.cumsum(var_exp)
#     print(f"{number_components} components exlain {cum_var_exp[number_components-1]*100: .1f} % of total variation")

#     # plotting
#     image_name = title_var.replace(" ", "") + ".png"
#     fig, ax = plt.subplots(figsize = (6,6))
#     x_vals = range(1, len(cum_var_exp)+1)

#     ax.bar(x_vals, var_exp,
#             alpha=1, align='center',
#             label='Individual', color = 'grey')

#     ax.step(x_vals, cum_var_exp, where = "mid", label='Cumulative', color = "black")

#     plt.xlim(x_lim)
#     plt.ylim([0,1])
#     plt.ylabel('Relative Variance Ratio')
#     plt.xlabel('Principal Components')
#     plt.legend(loc='best')
#     plt.grid(which = "both", axis = "y")

#     if title_var:
#         plt.title(title_var)

#     plt.tight_layout()

#     if save_path:
#         plt.savefig(save_path / image_name, facecolor='w', dpi = 100)

#     plt.show()

# def time_var_loading_plot(x_vals, y_vals, c_vals,
# x_label, c_label, title_var = None, save_fname = None, save_path = None):
 
#     fig, ax = plt.subplots(figsize = (10,8))
    
#     color_scale_lim = c_vals.abs().max()

#     im = ax.scatter(x_vals,
#     y_vals, c = c_vals,
#     cmap = "coolwarm_r", marker = 's', s = 200, alpha = 0.5,
#     edgecolor = None, vmin = -1*color_scale_lim, vmax = color_scale_lim)

#     fig.colorbar(im, ax=ax, label = c_label)
#     ax.set_ylabel("")
#     ax.set_xlabel(x_label)
#     ax.grid(which = "both", axis = "x")
      
#     if title_var:
#         plt.title(title_var)
        
#     plt.tight_layout()
    
#     if save_path:
#         if save_fname == None:
#             save_fname = title_var.replace(" ","")
#         else:
#             pass

#         image_name = save_path / (save_fname.replace(" ", "")+".png")
#         plt.savefig(image_name, facecolor='w')
    
#     plt.show()

# def parity_plot(y_val_train, y_val_train_pred,
# x_label, y_label, title_var, y_val_test = None, y_val_test_pred = None, save_path = None,
# alpha_var = 0.7, marker_s_var = 70, zero_axis_lim = True):
#     # plt.style.use('default')

#     fig, ax = plt.subplots(figsize=(7, 7))

#     ax.scatter(y_val_train,
#     y_val_train_pred, color = tab10_colors_list[0],
#     alpha=alpha_var, s = marker_s_var, label = f"Train")

#     if y_val_test is not None:
#         all_ys = list(y_val_train) + list(y_val_train_pred)+\
#         list(y_val_test_pred) + list(y_val_test)

#         ax.scatter(y_val_test,
#         y_val_test_pred, color = tab10_colors_list[1],
#         alpha=alpha_var, s = marker_s_var, label = f"Test")

#         plt.legend(loc = "upper left")
    
#     else:
#         all_ys = list(y_val_train) + list(y_val_train_pred)
    
#     if zero_axis_lim == False:
#         y_lims = [np.round(max(all_ys)*-1.1), np.round(max(all_ys)*1.1, 2)] # np.round(min(all_ys)*0.9, 2)
#     else:
#         y_lims = [0, np.round(max(all_ys)*1.1, 2)]

#     ax.plot(y_lims, y_lims, linestyle = "--", color = "grey")    

#     ax.set(xlabel=x_label, ylabel= y_label, title= title_var)
    
#     ax.set_ylim(y_lims)
#     ax.set_xlim(y_lims)

#     plt.grid(which = "both", axis = "both")
#     plt.tight_layout()

#     if save_path:
#         image_name = save_path / (title_var.replace(" ", "")+".png")
#         plt.savefig(image_name, facecolor='w')

#     plt.show()
