from sklearn.gaussian_process import GaussianProcessRegressor
import copy
"""
Update a Gaussian Process model with new X and y data (iteration 1+)
Also does not do hyperparameter search (optimizer=None).

:param gp: a given GP model
:param X: a pandas DataFrame of new training X, decoded
:param y: an array of new training y
:return: a GP model updated with the new training data.
"""
def update_model(gp, X, y):
    X_copy = X.copy()
    #must encode the discrete factors
    for factor in factors.factors:
        if factor[2] == "Ordinal" or factor[2] == "Categorical":
            #create mapping between levels
            levels = factor[1]
            name = factor[0]
            mapping = dict()
            for idx in range(len(levels)):
                level = levels[idx]
                mapping[level] = float(idx)
            X_copy[name] = X_copy[name].replace(mapping)
    X_copy_np = X_copy.to_numpy()

    #include old trained data from previous iterations
    all_X = np.vstack([gp.X_train_, X_copy_np])
    all_y = np.vstack([gp.y_train_, y])

    #change back into pandas DataFrame
    all_X_df = pd.DataFrame(all_X)
    
    #turn off hyperparameter searching
    gp.optimizer=None

    #fit the GP onto all the data and with no hyperparameter tuning
    gp.fit(all_X_df,all_y)
    return gp

"""
Initial train a GP on a decoded dataset, must encode the discrete factors first. 
Does hyperparameter search. 

:param gp: a given GP model
:param X_design: a pandas DataFrame containing the initial X design decoded
:param y: an array of new training y
:return: a GP model updated with the new training data.
"""
def train_model(gp, X, y, factors):
    X_copy = X.copy()
    #must encode discrete factors first
    for factor in factors.factors:
        if factor[2] == "Ordinal" or factor[2] == "Categorical":
            #create mapping between levels
            levels = factor[1]
            name = factor[0]
            mapping = dict()
            for idx in range(len(levels)):
                level = levels[idx]
                mapping[level] = float(idx)
            X_copy[name] = X_copy[name].replace(mapping)
    #print("Encoded discrete factors:\n",X_copy)
    #fit the GP onto all the data and with no hyperparameter tuning
    gp.fit(X_copy,y)
    return gp