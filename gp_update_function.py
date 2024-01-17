from sklearn.gaussian_process import GaussianProcessRegressor

"""
Update a Gaussian Process model with new X and y data (iteration 1+)
Also does not do hyperparameter search (optimizer=None).

:param gp: a given GP model
:param X: an array of new training X
:param y: an array of new training y
:return: a GP model updated with the new training data.
"""
def update_model(gp, X, y):
    #include old trained data from previous iterations
    all_X = np.vstack([gp.X_train_, X])
    all_y = np.vstack([gp.y_train_, y])
    
    #turn off hyperparameter searching
    gp.optimizer=None

    #fit the GP onto all the data and with no hyperparameter tuning
    gp.fit(all_X,all_y)
    return gp