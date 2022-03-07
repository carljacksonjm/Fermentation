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
# tab10_colors_list = sns.color_palette("tab10") + sns.color_palette("tab10")
# marker_list = ["o","v","D","x","X","<","8","s","^","p","P","*","h","H","+","d","|","_"]
# linestyle_list = ['-','--', '-.','-','--', '-.','-','--', '-.']

#%% 
def pls_sfs(train,labels, cv_folds = 5, score_metric = 'neg_root_mean_squared_error'):
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

def F_squarred(x, A):
    y = x**(A)
    return y

def diff_F_squrred(x, A):
    y = (A)*x**(A-1)
    return y

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
            cmap = "viridis_r", marker = 'o', s = 100, alpha = 0.7,linewidth = 1.5,
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


#%%

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
            cmap = "viridis_r", marker = 'o', s = 100, alpha = 0.7,linewidth = 1.5,
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

#%% PCR
# for y_col in y_cols[:]:
#     print("-----------------------\n",y_col)
#     y_col_short = y_col.split(" (")[0].title()#.replace(" ","")
#     y_units = y_col.split(" ")[-1]

#     pca_df_model = pca_df.dropna(subset = [y_col], axis = 0)
#     X_pcr = pca_df_model[pca_cols].to_numpy()
#     y = pca_df_model[y_col].to_numpy().reshape(-1, 1)
#     scaler_y = StandardScaler()
#     scaler_y.fit(y) # get mean and std dev for scaling
#     y_pcr = scaler_y.transform(y)

#     reg = LinearRegression()
#     y_pred_nonscale = cross_val_predict(reg, X_pcr, y_pcr, cv = cv_val)# retrieve the PCA step of the pipeline
#     y_pred = scaler_y.inverse_transform(y_pred_nonscale)
#     y_res = standardised_res(y[:,0], y_pred[:,0])

#     reg = LinearRegression()
#     reg.fit(X_pcr, y_pcr)
#     y_pred_nonscale_fit = reg.predict(X_pcr)
#     y_pred_fit = scaler_y.inverse_transform(y_pred_nonscale_fit)
#     y_res_fit = standardised_res(y[:,0], y_pred_fit[:,0])

#     r2_pred = r2_score(y, y_pred)
#     rmse_pred = abs(mean_squared_error(y, y_pred))**0.5

#     r2_fit = r2_score(y, y_pred_fit)
#     rmse_fit = abs(mean_squared_error(y, y_pred_fit))**0.5

#     r2_cvs = cross_val_score(reg, X_pcr, y_pcr,
#     cv = cv_val, scoring= "r2")
#     rmse_cvs = abs(cross_val_score(reg, X_pcr, y_pcr,
#     cv = cv_val, scoring= 'neg_root_mean_squared_error'))

#     parity_res_plots(y, y_pred_fit, y_res_fit,
#         y_col,
#         y_val_test = y, y_val_test_pred =  y_pred, y_val_test_res = y_res,
#         y_val_train_label = "Fitted", y_val_test_label = f"CV = {cv_val}",
#         save_fname = f"{y_col_short}_PCRParRes", save_path = pca_path,
#         zero_axis_lim = False)

#     reg.fit(X_pcr, y_pcr)   
#     df_coefs = pd.Series(data = reg.coef_[0,:], index = pca_cols, name = "Coefficients").T

#     contribution_chart(pca_exp_cols, df_coefs, "Coefficients", xlim = None,
#         save_path = pca_path, save_fname = f"{y_col_short}_PCRCoefs")

#     pcr_string = f"""
#     R2 CVS (CV = {cv_val}): {r2_cvs.mean(): .2f} ({r2_cvs.std(): .2f})
#     RMSE CVS (CV = {cv_val}): {rmse_cvs.mean(): .2f} ({rmse_cvs.std(): .2f})\n
#     R2 CVP (CV = {cv_val}): {r2_pred: .2f}
#     RMSE CVP (CV = {cv_val}): {rmse_pred: .2f} {y_units}\n    
#     R2 Fitted = {r2_fit: .2f}
#     RMSE Fitted = {rmse_fit: .2f} {y_units}
#     """
#     print(pcr_string)

#     pcr_fname = f"{y_col_short}_PCR.txt".replace(" ","")
#     save_to_txt(pcr_fname, pca_path, pcr_string)
