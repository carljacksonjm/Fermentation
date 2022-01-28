"""
base pump , contin / discrete 
some media better for growth than others?
optimal pcas for each stage
pc values for each stage
"""
#%% imports
# from re import S
from matplotlib.markers import MarkerStyle
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.pipeline import make_pipeline
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import RepeatedStratifiedKFold
# from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
import math
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import feature_selection
from sklearn import pipeline
import time
from scipy.optimize import curve_fit
from scipy.stats import linregress
# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import Imputer
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.metrics import make_scorer
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
# from sklearn.metrics import r2_score
from sklearn.ensemble import AdaBoostRegressor
# from sklearn.model_selection import cross_val_score
# from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
# import matplotlib.pyplot as plt
# import tflearn
# import tensorflow as tf
# import seaborn
# import warnings
# warnings.filterwarnings('ignore')

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

def Scatter3D(x_vals, y_vals, z_vals,
x_label, y_label, z_label,
c_val = None, c_label = None, x_rot = 30, marker_size = 170,
save_fname = None, save_path = None):
 
    fig = plt.figure(figsize = (6,6)) # dpi = 600
    ax = fig.add_subplot(projection = "3d")

    if c_label:

        im = ax.scatter(x_vals,
                    y_vals,
                    z_vals, c = c_val, 
                cmap = "cividis", marker = 'o',
                s = marker_size, alpha = 0.8, edgecolor = "black", label = c_label)
        
        fig.colorbar(im, ax = ax,label = f"{c_label}")# ax=ax, 
    else:
        im = ax.scatter(x_vals,
                    y_vals,
                    z_vals,  
                marker = 'o',
                s = marker_size, alpha = 0.5, edgecolor = jm_colors_list[0], color = jm_colors_list[0])
    ax.view_init(20, x_rot)
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

def get_r2(x_vals, y_vals):
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x_vals, y_vals)
    return slope, intercept, round(r_value**2,2)

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

def pca_biplot(x_vals, y_vals, x_label, y_label, c_val = None, c_label = None,
save_fname = None, save_path = None):
    
    fig, ax = plt.subplots(figsize = (10,6))

    x_lims = [x_vals.min()*1.1, x_vals.max()*1.1]
    y_lims = [y_vals.min()*1.1, y_vals.max()*1.1]

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

    ax.plot(x_lims,[0,0], linestyle = "--", color = "black", linewidth = 1.5)
    ax.plot([0,0], y_lims, linestyle = "--", color = "black", linewidth = 1.5)

    ax.set(ylabel = y_label, xlabel = x_label, xlim = x_lims, ylim = y_lims)
    plt.tight_layout()

    plt.grid(which = "both", axis = "both")
    
    if save_path and save_fname:
        image_name = save_path / (save_fname.replace(" ","") + ".png")
        
        plt.savefig(image_name, facecolor='w')
     
    plt.show()

#%% input variable
source_path = Path("C:/Users/JacksC01/OneDrive - Johnson Matthey/Documents/DigitalModelling/JMTCi4Fermentation/data")
output_path = Path("C:/Users/JacksC01/OneDrive - Johnson Matthey/Documents/DigitalModelling/JMTCi4Fermentation/figures")

source_fname = "batchwise_fermentation.csv"

cv_val = 7
pca_components = 3
#%% import of production summary and clean
df = pd.read_csv(source_path / source_fname)
df.set_index("USP", drop = True, inplace = True)

features = ['Volumetric Activity (U/ml)','Harvest WCW (g/l)',
 'Biomass Density (g/l)']

X_cols = list((set(df.columns) - set(features))) #- {"USP"})
X_cols_trends = [col for col in X_cols if ", t=" in col]
X_cols_ps = list((set(X_cols) - set(X_cols_trends)))
df.sort_values(by = ["Scale (l)"]+ features, ascending = True, inplace = True)
cols_remove_for_pca = [col for col in X_cols if ", t" in col]
X_cols_pcr = list(set(X_cols) - set(cols_remove_for_pca))

# data prep
pca_path = create_dir("PCA", output_path)
X = df[X_cols_pcr].to_numpy()
scaler_X = StandardScaler()#MinMaxScaler()
scaler_X.fit(X) # get mean and std dev for scaling
X_scaled = scaler_X.transform(X) # scaled

y = df[features].to_numpy()

usp_index = df.index.to_list()
print("Data ready.")
#%%
pca_var_explained(X_scaled, pca_components, x_lim = [0,8],
save_fname = "BWUPCAExplainedVariance", save_path = pca_path)

cov_mat = np.cov(X_scaled.T) # covariance matrix
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat) # decompose
tot = sum(eigen_vals)
var_exp = [np.real(i) / np.real(tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

loadings_pca, scores_pca, eigen_vals_pca = calibrate_pca(X_scaled, pca_components)
print(f"Loadings {loadings_pca.shape}\nScores {scores_pca.shape}\nEigen vals{eigen_vals_pca.shape}")

pca_cols = ["PC{0}".format(i) for i in range(1,scores_pca.shape[1]+1)]
pca_exp_cols = ["PC{0} ({1} %)".format(i, round(var_exp[i-1] * 100, 1)) for i in range(1, scores_pca.shape[1]+1)]

#%%
pca_df = pd.DataFrame(data = scores_pca, columns = pca_cols, index = usp_index)
# df_features = pd.DataFrame(data = y, columns = features, index = usp_index)
# pca_df = pd.merge(left = pca_df, right = df[["Scale (l)"] + features], left_index = True,
# right_index = True)
pca_df = pd.merge(left = pca_df, right = df[features], left_index = True,
right_index = True)

#pd.merge(left = pca_df, right = df_features, how = "left", left_index = True, right_index = True)

loadings_df = pd.DataFrame(loadings_pca, columns= pca_cols, index = X_cols_pcr)
loadings_df["Feature"] = loadings_df.index
loadings_df["Parameter"] = loadings_df["Feature"].apply(lambda x: x.rsplit(", t=", 2)[0])

df_timevar = loadings_df.drop(X_cols_ps, axis = 0) # drop non-process variables
df_timevar["Time (H)"] = df_timevar["Feature"].apply(lambda x: float(x.split(", t=")[-1])) # split time and parameter
# pivot by time and parameter
df_timevar_melt = pd.melt(df_timevar, id_vars=["Parameter", "Time (H)"], value_vars = pca_cols,
var_name="PC", value_name='LOADINGS', col_level=None, ignore_index=True)
df_timevar_melt["LOADINGS"] = df_timevar_melt["LOADINGS"].astype(float)
df_timevar_melt["LOADINGS ABS"] = df_timevar_melt["LOADINGS"].abs()
# df_timevar_melt.sort_values(by = "Time (H)", ascending= True, inplace = True)
df_timevar_melt.sort_values(by = ["Parameter", "Time (H)"], ascending= True, inplace = True)

# scatter plot of pcs vs features
pca_df["USP"] = pca_df.index
pca_df.reset_index(drop = True, inplace = True)
pca_df_melt = pd.melt(pca_df, id_vars = ["USP"]+features,
    var_name = "Component", value_name = "Value")

#%% pcr grid search
# all_r2 = []
# cv_val = 7
# scorer_var = 'neg_root_mean_squared_error'
# for y_col in features[:]:
#     print("-----------------------\n",y_col)
#     y_col_short = y_col.split(" (")[0].title()#.replace(" ","")
#     y_units = y_col.split(" ")[-1]
#     y_col_path = create_dir(y_col_short + "_PCA", output_path)

#     X_y_df = df[X_cols + [y_col]].dropna(how = "any", axis = 0)
#     y = X_y_df[y_col].to_numpy().reshape(-1, 1)
#     scaler_y = MinMaxScaler()
#     scaler_y.fit(y)
#     y_pcr = scaler_y.transform(y)

#     X = X_y_df[X_cols].to_numpy()
#     scaler_X = StandardScaler()
#     scaler_X.fit(X) # get mean and std dev for scaling
#     X_pcr = scaler_X.transform(X) # scale

#     t0 = time.time()
#     grid, pca, grid_params, grid_score = pcr(X_pcr, y_pcr, cv_folds = cv_val,
#         score_metric = scorer_var)
#     # pcr,

#     print(grid, pca, pcr, grid_params, grid_score)


#     run_time = (time.time() - t0) / 60
#     print(f"{run_time: .1f} mins")


#     grid_results = pd.DataFrame(data = grid.cv_results_)

#     # print(grid_results[["param_pca__n_components", "mean_test_score"]])
#     # results_features = grid_results.groupby("param_pca__n_components'")["rank_test_score"].min()
#     # results_features = grid_results.loc[grid_results["rank_test_score"].isin(results_features)]
#     # results_features.set_index("param_feat_selection__n_features_to_select", drop = True, inplace = True)
#     # results_features.sort_index(ascending=True, inplace = True)
#     # results_features["mean_test_score"] = results_features["mean_test_score"].abs()
#     # best_scores_std = results_features["std_test_score"].loc[results_features["rank_test_score"] == 1].abs().mean()
#     # # results_features["mean_test_score"] = scaler_y.inverse_transform(results_features["mean_test_score"].to_numpy())

#     scatter_line_plot(grid_results["param_pca__n_components"], grid_results["mean_test_score"]*-1,
#         "N Principal Components", f"RMSE (CV={cv_val})", y_std = grid_results["std_test_score"], y_lims = None,
#         y_scale = 'linear', save_fname = None, save_path = y_col_path) # f"{y_col_short}_PLSRMSE" grid_results["std_test_score"]

#%% pcr fitting
all_r2 = []
cv_val = 7

for y_col in ['Volumetric Activity (U/ml)', 'Biomass Density (g/l)']:
    print("-----------------------\n",y_col)
    y_col_short = y_col.split(" (")[0].title()#.replace(" ","")
    y_units = y_col.split(" ")[-1]
    # g = sns.FacetGrid(pca_df_melt, col="Component",
    #     col_wrap = 4, sharey = True, sharex = True, height = 5, aspect = 1)
    # g.map(sns.scatterplot, "Value", y_col, alpha = 0.75, s = 80,
    #     edgecolor = tab10_colors_list[0], color = tab10_colors_list[0])
    # for i, ax in enumerate(g.axes[:]):

    #     ax.set_xlabel(pca_cols[i])
    #     # ax.axhline(0,ls = "--", linewidth = 1, alpha = 0.5, color = "black")
    #     plt.tight_layout()
    #     ax.grid(which = "both", axis = 'both')

    # g.set(xlim = [-6,6],title = "")
    # # g.set_titles("")#col_template="{col_name}"
    # save_fname = f"{y_col_short}_PCScattering"
    # image_name = output_path / (save_fname.replace(" ", "")+".png")
    # plt.savefig(image_name, facecolor='w')
    # plt.show()

    # PCR
    pca_df_model = pca_df.dropna(subset = [y_col], axis = 0)

    X_pcr = pca_df_model[pca_cols].to_numpy()
    y = pca_df_model[y_col].to_numpy().reshape(-1, 1)
    scaler_y = MinMaxScaler()
    scaler_y.fit(y) # get mean and std dev for scaling
    y_pcr = scaler_y.transform(y)
    reg = LinearRegression()
    # y_pred_nonscale = reg.predict(X_pcr)
    y_pred_nonscale = cross_val_predict(reg, X_pcr, y_pcr, cv = cv_val)# retrieve the PCA step of the pipeline
    y_pred = scaler_y.inverse_transform(y_pred_nonscale)
    y_res = standardised_res(y[:,0], y_pred[:,0])

    reg = LinearRegression()
    reg.fit(X_pcr, y_pcr)
    y_pred_nonscale_fit = reg.predict(X_pcr)
    y_pred_fit = scaler_y.inverse_transform(y_pred_nonscale_fit)
    y_res_fit = standardised_res(y[:,0], y_pred_fit[:,0])

    r2_pred = r2_score(y, y_pred)
    rmse_pred = abs(mean_squared_error(y, y_pred))**0.5
    all_r2.append(r2_pred)

    r2_fit = r2_score(y, y_pred_fit)
    rmse_fit = abs(mean_squared_error(y, y_pred_fit))**0.5

    #reg = LinearRegression()
    r2_cvs = cross_val_score(reg, X_pcr, y_pcr,
    cv = cv_val, scoring= "r2")
    rmse_cvs = abs(cross_val_score(reg, X_pcr, y_pcr,
    cv = cv_val, scoring= 'neg_root_mean_squared_error'))

    parity_res_plots(y, y_pred_fit, y_res_fit,
        y_col,
        y_val_test = y, y_val_test_pred =  y_pred, y_val_test_res = y_res,
        y_val_train_label = "Fitted", y_val_test_label = f"CV = {cv_val}",
        save_fname = f"{y_col_short}_PCRParRes", save_path = pca_path,
        zero_axis_lim = False)

    reg.fit(X_pcr, y_pcr)   
    df_coefs = pd.Series(data = reg.coef_[0,:], index = pca_cols, name = "Coefficients").T
    # df_coefs = df_coefs.sort_values(key = abs, ascending = False)

    #df_coefs.index
    contribution_chart(pca_exp_cols, df_coefs, "Coefficients", xlim = None,
        save_path = pca_path, save_fname = f"{y_col_short}_PCRCoefs")

    pcr_string = f"""
    R2 CVS (CV = {cv_val}): {r2_cvs.mean(): .2f} ({r2_cvs.std(): .2f})
    RMSE CVS (CV = {cv_val}): {rmse_cvs.mean(): .2f} ({rmse_cvs.std(): .2f})\n
    R2 CVP (CV = {cv_val}): {r2_pred: .2f}
    RMSE CVP (CV = {cv_val}): {rmse_pred: .2f} {y_units}\n    
    R2 Fitted = {r2_fit: .2f}
    RMSE Fitted = {rmse_fit: .2f} {y_units}
    """
    print(pcr_string)

    pcr_fname = f"{y_col_short}_PCR.txt".replace(" ","")
    save_to_txt(pcr_fname, pca_path, pcr_string)

print(all_r2)

#%% plot time variant loadings 
# max_loading = (loadings_df[pca_cols].abs().max().max())
# loadings_range = [max_loading*-1, max_loading]
# c_range = [max_loading*-0.5, max_loading*0.5]
# for pc in pca_cols:
#     print("\n------------------\n",pc)
    # time_var_loading_plot(df_timevar_melt["Time (H)"].loc[(df_timevar_melt["PC"] == pc)],
    #     df_timevar_melt["Parameter"].loc[(df_timevar_melt["PC"] == pc)],
    #     df_timevar_melt["LOADINGS"].loc[(df_timevar_melt["PC"] == pc)],
    #     "Time (h)", "Loading", clims = c_range, save_fname = f"VarLoadings{pc}",
    #     save_path = pca_path)

#%%
df_timeinvar = loadings_df.loc[loadings_df.index.isin(X_cols_ps)]
df_timeinvar.sort_values(by = ["Feature"], ascending = True, inplace = True)
df_timeinvar_melt = pd.melt(df_timeinvar,
id_vars=["Parameter"],
value_vars=pca_cols,
var_name="PC",
value_name="Loading", ignore_index = False)
# contribution_chart(df_timeinvar.Feature, df_timeinvar[pc], "Loadings", xlim = loadings_range,
#         save_path = pca_path, save_fname = f"InvLoadings{pc}")

ax = sns.catplot(kind = "bar",data = df_timeinvar_melt,
    y = "Parameter", x = "Loading", col = "PC",
    hue = "PC", legend = False, sharey = True,palette = tab10_colors_list, dodge = False,
    height = 8, aspect = 1.25)

ax.set_titles(col_template="{col_name}")
ax.set(ylabel = "")

image_name = pca_path / ("PCLoadings".replace(" ", "")+".png")
plt.savefig(image_name, facecolor='w')
plt.show()

#%% visualisation pca
def lin_reg(x_vals, y_vals):
    m, c, r_value, p_value, std_err = linregress(x_vals, y_vals)
    x_func = np.linspace(x_vals.min(), x_vals.max(), 50)
    y_func = m * x_func + c
    y_pred = m * x_vals + c
    r2_pred = r2_score(y_vals, y_pred)
    rmse_pred = abs(mean_squared_error(y_vals, y_pred))**0.5

    return m, c, std_err, r2_pred, rmse_pred, x_func, y_func

# scatter_2D(pca_df["PC2"], pca_df['Volumetric Activity (U/ml)'],
# x_label = pca_exp_cols[1], y_label = 'Volumetric Activity (U/ml)', save_fname = f"VolumetricActivity_PC2", save_path = pca_path)

dydx, inter, std_error, r2, rmse, x_range, y_range = lin_reg(pca_df["PC2"], pca_df['Volumetric Activity (U/ml)'])
confidence_interval = 1.96*std_error
y_conf = np.array([y_range - confidence_interval, y_range + confidence_interval]).T
print(f"R2 {r2:.2f}, RMSE {rmse: .2f}")

fig, ax = plt.subplots(figsize = (10,6))
ax.plot(x_range, y_range,
    linestyle = "--", color = "black")
ax.fill_between(x_range, y_conf[:,1], y_conf[:, 0],
    color = "grey", alpha = 0.2, label = "95% CI")
ax.scatter(pca_df["PC2"], pca_df['Volumetric Activity (U/ml)'],
        s = 60, linewidth = 1.4,
        edgecolor = "black", facecolor = "none")
ax.set(ylabel = 'Volumetric Activity (U/ml)', xlabel = pca_exp_cols[1])
plt.tight_layout()
image_name = pca_path / ("VolumetricActivity_PC2".replace(" ", "")+".png")
plt.savefig(image_name, facecolor='w')
plt.show()

pca_biplot(pca_df["PC1"], pca_df["PC3"], pca_exp_cols[0], pca_exp_cols[2],
c_val = pca_df['Biomass Density (g/l)'], c_label = "Biomass Density (g/l)",
save_fname = f"BiomassDensity_PC1PC3", save_path = pca_path)

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

from scipy import stats
alpha_stats = 0.05
CI = [n_exp + std_err*t for t in stats.t.interval(alpha_stats/2, len(x_c_y[:,0])-2)]
halfwidth = std_err*stats.t.interval(alpha_stats/2, len(x_c_y[:,0])-2)[1]
n_exp_str = r'$\alpha$={:.2f}($\pm${:.3f})'.format(n_exp, halfwidth)

# std_error_slope = se_slope(np.log10(x_c_y[:,2]), y_pred_exp, x_c_y[:,0])
# slope_error = 1.96 * std_error_slope
# max_slope = n_exp + slope_error
# print(slope_error)

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


#%% vol activity plots
import scipy




#%%
# feature_combs = ['Base (2h MA), t=12.0',"Power (W/m3), t=12.0",'Base (2h MA), t=9.0',"Power (W/m3), t=9.0"]

# print("R2 with Volumetric Activity")
# for f in feature_combs:
#     print(f, " ", get_r2(X_scaled[:,X_cols.index(f)], y_scaled[:,features.index('Volumetric Activity (U/ml)')]))

# plt.rcParams.update({'font.size': 12})

# fig = plt.figure(figsize=(12,6))
# ax1 = fig.add_subplot(1, 2, 2, projection='3d')
# ax1.scatter(X_scaled[:,X_cols.index('Base (2h MA), t=12.0')],
#     X_scaled[:,X_cols.index("Power (W/m3), t=12.0")],
#     y_scaled[:,features.index('Volumetric Activity (U/ml)')],
#     label = "12 hours",
#     color = tab10_colors_list[1], edgecolor = tab10_colors_list[1], marker = 'o', s = 150, alpha = 0.5, linewidth = 1.5)

# ax2 = fig.add_subplot(1, 2, 1, projection='3d')
# ax2.scatter(X_scaled[:,X_cols.index('Base (2h MA), t=9.0')],
#     X_scaled[:,X_cols.index("Power (W/m3), t=9.0")],
#     y_scaled[:,features.index('Volumetric Activity (U/ml)')],
#     label = "9 hours",
#     color = tab10_colors_list[0], edgecolor = tab10_colors_list[0], marker = 'o', s = 150, alpha = 0.5, linewidth = 1.5)

# handles, labels = [], []
# for ax in [ax1, ax2]:
#     ax.set(xlabel = "Base (2h MA)", ylabel = "Specific Power", zlabel = 'Volumetric Activity (U/ml)')
#     h, l = ax.get_legend_handles_labels()
#     handles.append(h[0])
#     labels.append(l[0])

# ax2.legend(handles= handles, labels = labels, loc = "upper left",
#     framealpha = 1, shadow = False)

# image_name = output_path / ("VolumetricActivity_BasePowert9t12".replace(" ", "")+".png")
# plt.savefig(image_name, facecolor='w')
# plt.show()

# power (W/m3), t=9.0
# antifoam,  X[:,X_cols.index('antifoam')] == 1]
# scale (l) X[:,X_cols.index('antifoam')] == 42]
# base ?
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

fig, ax = plt.subplots(figsize = (10,6))

ax.scatter(
    X_scaled[(X[:,X_cols.index("Scale (l)")] == 1) & (X[:,X_cols.index("Antifoam")] == 0), X_cols.index("Power (W/m3), t=9.0")],
    y_scaled[(X[:,X_cols.index("Scale (l)")] == 1) & (X[:,X_cols.index("Antifoam")] == 0), features.index('Volumetric Activity (U/ml)')],
    marker = marker_list[0], facecolor = "none", label = "1 l",
    color = tab10_colors_list[0], s = 150, alpha = 1, linewidth = 2)
ax.scatter(
    X_scaled[(X[:,X_cols.index("Scale (l)")] == 1) & (X[:,X_cols.index("Antifoam")] == 1), X_cols.index("Power (W/m3), t=9.0")],
    y_scaled[(X[:,X_cols.index("Scale (l)")] == 1) & (X[:,X_cols.index("Antifoam")] == 1), features.index('Volumetric Activity (U/ml)')],
    marker = marker_list[0], facecolor = "none", label = "1 l (A)",
    color = tab10_colors_list[1], s = 150, alpha = 1, linewidth = 2)

ax.scatter(
    X_scaled[(X[:,X_cols.index("Scale (l)")] == 42) & (X[:,X_cols.index("Antifoam")] == 0), X_cols.index("Power (W/m3), t=9.0")],
    y_scaled[(X[:,X_cols.index("Scale (l)")] == 42) & (X[:,X_cols.index("Antifoam")] == 0), features.index('Volumetric Activity (U/ml)')],
    marker = marker_list[1], facecolor = "none", label = "42 l",
    color = tab10_colors_list[2], s = 150, alpha = 1, linewidth = 2)

ax.scatter(
    X_scaled[(X[:,X_cols.index("Scale (l)")] == 42) & (X[:,X_cols.index("Antifoam")] == 1), X_cols.index("Power (W/m3), t=9.0")],
    y_scaled[(X[:,X_cols.index("Scale (l)")] == 42) & (X[:,X_cols.index("Antifoam")] == 1), features.index('Volumetric Activity (U/ml)')],
    marker = marker_list[1], facecolor = "none", label = "42 l (A)",
    color = tab10_colors_list[3], s = 150, alpha = 1, linewidth = 2)

plt.legend(title = "Scale (Antifoam)", loc = "best",
    framealpha = 1, shadow = False)
ax.set(xlabel = "Power, t=9.0", ylabel = "Volumetric Activity", xlim = [-0.1,1.1], ylim = [-0.1,1.1])
# ax.grid(which = "both", axis = "both")

plt.tight_layout()
image_name = output_path / ("VolumetricActivity_Power9".replace(" ", "")+".png")
plt.savefig(image_name, facecolor='w')
plt.show()

scatter_2D(X_scaled[:, X_cols.index("Power (W/m3), t=12.0")],
y_scaled[:, features.index('Volumetric Activity (U/ml)')], "x", "y")

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

#%%
X_cols = sorted(X_cols)
# X_cols_manip = [
#  'Antibiotic_Kan',
#  'Feed Media_F04',
#  'Feed Seq Red at Induct (%)',
#  'Feed Seq Step_Steps',
#  'Feed Total, t=12.0',
#  'Feed Total, t=22.0',
#  'Feed Total, t=27.0',
#  'Feed Total, t=35.0',
#  'Feed Total, t=4.0',
#  'Feed Total, t=45.0',
#  'Feed Total, t=9.0',
#  'Glycerol ID_MGS-0467',
#  'Induction Temperature (°C)',
#  'Induction pH',
#  'Production Media_gMSMK',
#  'Production Media_gMSMK mod1',
#  'Production Media_gMSMK mod2',
#  'Scale (l)',
#  'Temperature (°C), t=12.0',
#  'Temperature (°C), t=22.0',
#  'Temperature (°C), t=27.0',
#  'Temperature (°C), t=35.0',
#  'Temperature (°C), t=4.0',
#  'Temperature (°C), t=45.0',
#  'Temperature (°C), t=9.0',
#  'pH, t=12.0',
#  'pH, t=22.0',
#  'pH, t=27.0',
#  'pH, t=35.0',
#  'pH, t=4.0',
#  'pH, t=45.0',
#  'pH, t=9.0']

# X_cols_manip = [
#  'Antibiotic_Kan',
#  'Feed Media_F04',
#  'Feed Seq Red at Induct (%)',
#  'Feed Seq Step_Steps',
#  'Feed Total, t=12.0',
#  #'Feed Total, t=22.0',
#  'Feed Total, t=27.0',
#  'Feed Total, t=35.0',
#  #'Feed Total, t=4.0',
#  #'Feed Total, t=45.0',
#  #'Feed Total, t=9.0',
#  'Glycerol ID_MGS-0467',
#  'Induction Temperature (°C)',
#  #'Induction pH',
#  'Production Media_gMSMK',
#  'Production Media_gMSMK mod1',
#  'Production Media_gMSMK mod2',
#  'Scale (l)']

#%%

#%% SFS-PLS
def pls_rfe(train,labels, cv_folds = 5, score_metric = 'neg_root_mean_squared_error'):
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
  
    def rfe_grid_search(r_model, r_params):
        
        r = GridSearchCV(r_model, r_params,
            scoring = 'neg_root_mean_squared_error')
        r.fit(train, labels)
        
        return r#, r.best_estimator_, r.best_params_
    
    #"r2"
    est = PLSRegression()
    selector = feature_selection.SequentialFeatureSelector(est, scoring = score_metric)#feature_selection.RFE(est, step = 1)
    pipe_params = [('feat_selection',selector),('clf', est)]
    pipe = pipeline.Pipeline(pipe_params)

    clf_parameters = {'feat_selection__n_features_to_select': list(range(1,6)),
    'clf__n_components': list(range(1,4))}

    sfs_grid = GridSearchCV(pipe, clf_parameters,
            scoring = score_metric, cv = cv_val) # 
    sfs_grid.fit(train, labels)

    best_pls = sfs_grid.best_estimator_['clf']
    best_featselect = sfs_grid.best_estimator_['feat_selection']
    best_params = sfs_grid.best_params_
    best_score = sfs_grid.best_score_ 

    print("Grid search complete.")
        
    return sfs_grid, best_featselect, best_pls, best_params, best_score

#'Biomass Density (g/l)'"Volumetric Activity (U/ml)""Harvest WCW (g/l)"
X_cols_new = X_cols #X_cols_manip#X_cols

scorer_var = 'neg_root_mean_squared_error'#'mean_squared_error'#'neg_root_mean_squared_error'#"r2"
cv_val = 7

n_features = len(X_cols_new)
for y_col in features[0:1]:
    y_col_short = y_col.split(" (")[0].title()#.replace(" ","")
    y_units = y_col.split(" ")[-1]
    y_col_path = create_dir(y_col_short + "_" + scorer_var, output_path)

    X_y_df = df[X_cols_new + [y_col]].dropna(how = "any", axis = 0)
    y = X_y_df[y_col].to_numpy().reshape(-1, 1)
    scaler_y = MinMaxScaler()
    scaler_y.fit(y)
    y_pls = scaler_y.transform(y)

    X = X_y_df[X_cols_new].to_numpy()
    scaler_X = MinMaxScaler()
    scaler_X.fit(X) # get mean and std dev for scaling
    X_pls = scaler_X.transform(X) # scale

    n_obs = len(y)
    print(f"\n\n\n-------\n{n_features} features, {n_obs} datapoints")

    t0 = time.time()
    grid, featselect, pls, grid_params, grid_score = pls_rfe(X_pls, y_pls, cv_folds = cv_val,
        score_metric = scorer_var)
    run_time = (time.time() - t0) / 60
    print(f"{run_time: .1f} mins")

    pls_coef = pls.coef_
    pls_loadings = pls.x_loadings_
    featselect_sup = featselect.support_
    grid_score = abs(grid_score)

    grid_results = pd.DataFrame(data = grid.cv_results_)
    results_features = grid_results.groupby("param_feat_selection__n_features_to_select")["rank_test_score"].min()
    results_features = grid_results.loc[grid_results["rank_test_score"].isin(results_features)]
    results_features.set_index("param_feat_selection__n_features_to_select", drop = True, inplace = True)
    results_features.sort_index(ascending=True, inplace = True)
    results_features["mean_test_score"] = results_features["mean_test_score"].abs()
    best_scores_std = results_features["std_test_score"].loc[results_features["rank_test_score"] == 1].abs().mean()
    # results_features["mean_test_score"] = scaler_y.inverse_transform(results_features["mean_test_score"].to_numpy())

    scatter_line_plot(results_features.index, results_features["mean_test_score"],
        "N Features", f"RMSE (CV={cv_val})", y_std = results_features["std_test_score"], y_lims = None,
        y_scale = 'linear', save_fname = f"{y_col_short}_PLSRMSE", save_path = y_col_path)

    df_coefs = pd.DataFrame(data = [X_cols_new, featselect_sup], index = ["Parameter", "Selected"]).T
    df_coefs = df_coefs.loc[df_coefs["Selected"] == True]
    X_cols_sfs = list(df_coefs["Parameter"]) # selected features
    X_pls_sfs = X_pls[:, featselect_sup == True]
    df_coefs["Coefficient"] = pls_coef[:,0]
    df_coefs = df_coefs.sort_values(by = ["Coefficient"], key = abs, ascending = False)
    #df_coefs = df_coefs.loc[df_coefs["Coefficient"].abs() > 0.05]
    coefs_range = [-1,1]#[coefs_range*-1, coefs_range]
    # contribution_chart(df_coefs["Parameter"], df_coefs["Coefficient"], "Coefficient", xlim = coefs_range,
    #     save_path = output_path, save_fname = f"{y_col_short}_PLSCoefs{n_features}")

    # loadings df
    df_loadings = pd.DataFrame(data = pls_loadings, columns = ["PLS"+str(i+1) for i in range(len(pls_loadings.T))])
    df_loadings["Parameter"] = X_cols_sfs
    df_loadings_melt = pd.melt(df_loadings, id_vars = ["Parameter"], var_name = "Component", value_name = "Loading")
    df_loadings_melt = df_loadings_melt.loc[df_loadings_melt["Loading"].abs() >0.025]
    df_loadings["Time (h)"] = df_loadings["Parameter"].apply(lambda x: x.split(", t=")[-1])

    col_wrap_var = 1 if len(pls_loadings.T) == 1 else 2
    g = sns.catplot(kind = "bar", data = df_loadings_melt, y = "Parameter", x = "Loading", col = "Component",
    orient = "h", hue = "Parameter", palette = tab10_colors_list, dodge = False, edgecolor = "black", aspect = 2, col_wrap = col_wrap_var, legend = False)
    g.set_axis_labels("Loading","")
    g.set_titles("{col_name}")
    g.tight_layout()
    # plt.grid(which = "both", axis = 'x')
    save_fname = f"{y_col_short}_PLSLoadings"
    image_name = y_col_path / (save_fname.replace(" ", "")+".png")
    plt.savefig(image_name, facecolor='w')
    plt.show()

    # scatter important parameters
    top_params = list(df_coefs["Parameter"][:])
    plt.rcParams.update({'font.size': 14})
    subplots_scatter(X_y_df, top_params, y_col, n_cols = 4,
            title_var = None, save_fname = f"{y_col_short}_PLSScatterParams", save_path = y_col_path)
    plt.rcParams.update({'font.size': 16})

    y_pred_nonscale = pls.predict(X_pls_sfs)#cross_val_predict(pls, X_pls_sfs, y_pls, cv = cv_val)
    y_pred = scaler_y.inverse_transform(y_pred_nonscale)
    y_res = standardised_res(y[:,0], y_pred[:,0])

    r2_s = r2_score(y, y_pred)
    mse_s = mean_squared_error(y, y_pred)
    rmse_s = (abs(mse_s))**0.5

    r2_scores = cross_val_score(pls, X_pls_sfs, y_pls, cv = cv_val, scoring= "r2")#.mean()
    rmse_scores = abs(cross_val_score(pls, X_pls_sfs, y_pls, cv = cv_val, scoring= 'neg_root_mean_squared_error'))

    parity_res_plots(y, y_pred, y_res,
        y_col, y_val_test = None, y_val_test_pred = None,
        save_fname = f"{y_col_short}_PLSParRes", save_path = y_col_path,
        zero_axis_lim = False)

    pls_string = f"""Best model: {grid_params}
    Features: {X_cols_sfs}
    Best RMSE grid search: {grid_score: .2f} ({best_scores_std: .2f})
    R2 CV = {cv_val}: {r2_scores.mean(): .2f} ({r2_scores.std(): .2f})
    RMSE CV = {cv_val}: {rmse_scores.mean(): .2f} ({rmse_scores.std(): .2f})
    R2 = {r2_s: .2f}
    RMSE = {rmse_s: .2f} {y_units}
    MSE = {mse_s: .2f} {y_units}**2
    """
    print(pls_string)
    pls_fname = f"{y_col_short}_PLS.txt".replace(" ","")
    save_to_txt(pls_fname, y_col_path, pls_string)

    X_latents = pls.transform(X_pls_sfs)

    if grid_params['clf__n_components']==1:
            
        fig, ax = plt.subplots(figsize = (8,6))

        # axis_lims = [axis_lim*-1, axis_lim]
        im = ax.scatter(X_latents[:,0], y_pls, s = 80, facecolor = "none",
        edgecolor = "black")
        #fig.colorbar(im, ax=ax, label = y_col_short)
        #ax.plot(axis_lims, [0,0], color = "black", linewidth = 1, linestyle = "--")
        #ax.plot([0,0],axis_lims, color = "black", linewidth = 1, linestyle = "--")

        ax.set(#xlim = axis_lims, ylim = axis_lims,
        xlabel = f"$LV{i+1}$", ylabel = y_col_short)
        plt.grid(which = "both", axis = "both")

        plt.tight_layout()
        image_name = y_col_path / (f"{y_col_short}_LatentVars{i}".replace(" ", "")+".png")
        plt.savefig(image_name, facecolor='w')
        plt.show()
        #axis_lim = np.ceil(abs(X_latents).max())
        #axis_lims = [axis_lim*-1, axis_lim]

    else:

        for i in range(X_latents.shape[1]-1):
            
            pca_biplot(X_latents[:,i], X_latents[:,i+1], f"LV{i+1}", f"LV{i+2}", c_val = y_pls,
            c_label = y_col_short,
            save_fname = f"{y_col_short}_LatentVars{i+1}", save_path = y_col_path)

            # fig, ax = plt.subplots(figsize = (8,6))

            # # axis_lims = [axis_lim*-1, axis_lim]
            # im = ax.scatter(X_latents[:,i], X_latents[:,i+1],c = y_pls, s = 80, 
            # alpha = 0.6, edgecolor = "black")
            # fig.colorbar(im, ax=ax, label = y_col_short)
            # #ax.plot(axis_lims, [0,0], color = "black", linewidth = 1, linestyle = "--")
            # #ax.plot([0,0],axis_lims, color = "black", linewidth = 1, linestyle = "--")

            # ax.set(#xlim = axis_lims, ylim = axis_lims,
            # xlabel = f"$LV{i+1}$", ylabel = f"$LV{i+2}$")
            # plt.grid(which = "both", axis = "both")

            # plt.tight_layout()
            # image_name = y_col_path / (f"{y_col_short}_LatentVars{i+1}".replace(" ", "")+".png")
            # plt.savefig(image_name, facecolor='w')
            # plt.show()

            # del X_latents

# plt.plot(results_features.index, results_features*-1, marker = "o", color = "black")
# plt.ylabel("RMSE")
# plt.xlabel("N Features")
# # plt.ylim([-5,1])
# plt.grid(which = "both", axis = "both")
# plt.tight_layout()
# plt.show()

# screen ml
# df_nrmse = lets_try(X_pls, y_pls, cv_folds=cv_val)
# sns_barplot(df_nrmse, "RMSE","Method", orient_var = "h",
#     title_var = None, save_fname = f"{y_col_short}ModelScreening", save_path = None)#output_path

#%%




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