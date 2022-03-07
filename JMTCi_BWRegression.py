"""
base pump , contin / discrete 
some media better for growth than others?
optimal pcas for each stage
pc values for each stage
"""
#%% imports
# from re import S
from JMTCi_Functions import DoPickle, create_dir, save_to_txt, biplot
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import math
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import feature_selection

import time
from sklearn import pipeline
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
plt.rcParams.update({'font.size': 16})
tab10_colors_list = sns.color_palette("tab10") + sns.color_palette("tab10")

#%% functions
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

    ax.set(xlabel = x_label, ylabel = y_label, ylim = y_lims, xlim = x_lims, yscale = y_scale)

    ax.grid(which = "both", axis = "both")
    fig.tight_layout()

    if save_path and save_fname:
        image_name = save_path / (save_fname.replace(" ", "")+".png")
        plt.savefig(image_name, facecolor='w')
     
    plt.show()

def pls_sfs(train,labels, cv_folds = 5, score_metric = 'neg_root_mean_squared_error', max_features = 10, max_pls_comps = 3):
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

    clf_parameters = {'feat_selection__n_features_to_select': list(range(1, max_features+1)),
    'clf__n_components': list(range(1,max_pls_comps+1))}

    sfs_grid = GridSearchCV(pipe, clf_parameters,
            scoring = score_metric, cv = cv_folds) # 
    sfs_grid.fit(train, labels)

    best_pls = sfs_grid.best_estimator_['clf']
    best_featselect = sfs_grid.best_estimator_['feat_selection']
    best_params = sfs_grid.best_params_
    best_score = sfs_grid.best_score_ 

    print("Grid search complete.")
        
    return sfs_grid, best_featselect, best_pls, best_params, best_score

#%% input variables
source_path = Path("C:/Users/JacksC01/OneDrive - Johnson Matthey/Documents/DigitalModelling/JMTCi4Fermentation/data")
output_path = Path("C:/Users/JacksC01/OneDrive - Johnson Matthey/Documents/DigitalModelling/JMTCi4Fermentation/figures")

source_fname = "batchwise_fermentation.csv"
col_name_fname = "AllColsDict.pkl"

scorer_var = 'neg_root_mean_squared_error'#
scorer_label = "NRMSE"
cv_val = LeaveOneOut()#7
pls_max_features = 7
pls_max_comps = 2

#%% import data and create datasets
all_cols_dict = DoPickle(col_name_fname).pickle_load()
y_cols = all_cols_dict['y_cols']
X_cols_tvar = all_cols_dict['TimeVarParams']
X_cols_tinvar = all_cols_dict['TimeInvarParams']
X_cols = X_cols_tinvar + X_cols_tvar

process_params = all_cols_dict["ProcessParams"]

df = pd.read_csv(source_path / source_fname)
df.set_index("USP", drop = True, inplace = True)
df.sort_values(by = ["Scale (l)"]+ y_cols, ascending = True, inplace = True)

# remove Fr since cross-correl with Power
process_params.remove("log(Fr)")

#%% choose timeinvar points
X_cols_model = X_cols_tinvar.copy()
timeline_time_dict = {
    ", tb=":[1.0,3.0],
    ", tf=":[-1.0,3.0],
    ", ti=":[-1.0,3.0,21.0]
    }


for key, vals in timeline_time_dict.items():
    for val in vals:

        [X_cols_model.append(f"{col}{key}{val}") for col in process_params]

X_cols_model.sort()
# X_cols_model = X_cols_tinvar.copy()

#%% SFS-PLS
n_features = len(X_cols_model)
y_col_preds = {}
y_col_latents = {}
y_col_grid_search = {}

for y_col in ["Harvest WCW (g/l)", "Volumetric Activity (U/ml)"]:#y_cols[1:]:#
    print(f"\n\n\n-------\n{y_col}")
    y_col_short = y_col.split(" (")[0].title()#.replace(" ","")
    y_units = y_col.split(" ")[-1]
    y_col_path = create_dir(y_col_short + "_" + scorer_var, output_path)

    X_y_df = df[X_cols_model + [y_col]].dropna(how = "any", axis = 0)
    y = X_y_df[y_col].to_numpy().reshape(-1, 1)
    scaler_y = MinMaxScaler()
    scaler_y.fit(y)
    y_scaled = scaler_y.transform(y)

    X = X_y_df[X_cols_model].to_numpy()
    scaler_X = MinMaxScaler()
    scaler_var = scaler_X
    scaler_X.fit(X) # get mean and std dev for scaling
    X_scaled = scaler_X.transform(X) # scale

    n_obs = len(y)
    print(f"{n_features} features, {n_obs} datapoints")

    t0 = time.time()
    grid, featselect, pls, grid_params, grid_score = pls_sfs(X_scaled, y_scaled, cv_folds = cv_val, 
    score_metric = scorer_var, max_features = pls_max_features, max_pls_comps =  pls_max_comps)
    run_time = (time.time() - t0) / 60
    print(f"{run_time: .1f} mins")

    y_col_grid_search[y_col] = grid, featselect, pls, grid_params, grid_score

    pls_coef = pls.coef_
    pls_loadings = pls.x_loadings_
    featselect_sup = featselect.support_
    # grid_score = grid_score

    grid_results = pd.DataFrame(data = grid.cv_results_)
    results_features = grid_results.groupby("param_feat_selection__n_features_to_select")["rank_test_score"].max()
    results_features = grid_results.loc[grid_results["rank_test_score"].isin(results_features)]
    results_features.set_index("param_feat_selection__n_features_to_select", drop = True, inplace = True)
    results_features.sort_index(ascending=True, inplace = True)
    results_features["mean_test_score"] = results_features["mean_test_score"]
    best_scores_std = results_features["std_test_score"].loc[results_features["rank_test_score"] == 1].abs().mean()
    # results_features["mean_test_score"] = scaler_y.inverse_transform(results_features["mean_test_score"].to_numpy())

    scatter_line_plot(results_features.index, results_features["mean_test_score"],
        "N Features", f"{scorer_label} (LOOCV)", y_std = results_features["std_test_score"], y_lims = [-2,1],
        y_scale = 'linear', save_fname = f"{y_col_short}_{scorer_label}_features", save_path = y_col_path)

    df_coefs = pd.DataFrame(data = [X_cols_model, featselect_sup], index = ["Parameter", "Selected"]).T
    df_coefs = df_coefs.loc[df_coefs["Selected"] == True]
    X_cols_sfs = list(df_coefs["Parameter"])
    X_pls_sfs = X_scaled[:, featselect_sup == True]
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

    y_pred_scaled = pls.predict(X_pls_sfs)#cross_val_predict(pls, X_pls_sfs, y_scaled, cv = cv_val)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_res = standardised_res(y[:,0], y_pred[:,0])

    np_y = np.hstack((y, y_scaled, y_pred_scaled, y_pred, y_res.reshape(-1,1)))
    y_col_preds[y_col] = pd.DataFrame(data = np_y, columns = ["y", "y_scaled", "y_pred_scaled", "y_pred", "y_res"])

    r2_s = r2_score(y_scaled, y_pred_scaled) # y_scaled,y_pred_scaled 
    mse_s = mean_squared_error(y_scaled, y_pred_scaled)# y, y_pred
    rmse_s = (abs(mse_s))**0.5

    r2_scores = cross_val_score(pls, X_pls_sfs, y_scaled, cv = cv_val, scoring= "r2")#.mean()
    rmse_scores = abs(cross_val_score(pls, X_pls_sfs, y_scaled, cv = cv_val, scoring= 'neg_root_mean_squared_error'))

    parity_res_plots(y, y_pred, y_res, y_col, y_val_test = None, y_val_test_pred = None,
        save_fname = f"{y_col_short}_PLSParRes", save_path = y_col_path, zero_axis_lim = False)

    pls_string = f"""{y_col_short}
    {scaler_var}
    ---
    Best model:
    {grid_params}
    ---
    Features:
    {X_cols_sfs}
    ---
    {scorer_label} grid search: {grid_score: .2f} ({best_scores_std: .2f})
    R2 CV = {cv_val}: {r2_scores.mean(): .2f} ({r2_scores.std(): .2f})
    RMSE CV = {cv_val}: {rmse_scores.mean(): .2f} ({rmse_scores.std(): .2f})
    ---
    R2 = {r2_s: .2f}
    RMSE = {rmse_s: .2f}
    MSE = {mse_s: .2f}
    """
    print(pls_string)
    pls_fname = f"{y_col_short}_PLS.txt".replace(" ","")
    save_to_txt(pls_fname, y_col_path, pls_string)

    X_latents = pls.transform(X_pls_sfs)
    y_col_latents[y_col] = [X_latents, df_loadings_melt]

    if grid_params['clf__n_components']==1:
            
        fig, ax = plt.subplots(figsize = (8,6))
        im = ax.scatter(X_latents[:,0], y_scaled, s = 80, facecolor = "none",
        edgecolor = "black")
        ax.set(xlabel = f"$LV1$", ylabel = y_col_short)
        plt.grid(which = "both", axis = "both")
        plt.tight_layout()
        image_name = y_col_path / (f"{y_col_short}_LatentVars1".replace(" ", "")+".png")
        plt.savefig(image_name, facecolor='w')
        plt.show()

    else:

        for i in range(X_latents.shape[1]-1):

            biplot(X_latents[:,i], X_latents[:,i+1], f"LV{i+1}", f"LV{i+2}", c_val = y_scaled,
            c_label = y_col_short,save_fname = f"{y_col_short}_LatentVars{i+1}", save_path = y_col_path)

    del X_latents

#%%
print("Done.")

#%%
DoPickle("y_col_pres.pkl").pickle_save(y_col_preds)
DoPickle("y_col_latents.pkl").pickle_save(y_col_latents)

# %%
