from sklearn.gaussian_process import GaussianProcessRegressor
import copy
from InitStrategies import *
from FactorSet import *
"""
Update a Gaussian Process model with new X and y data (iteration 1+)
Also does not do hyperparameter search (optimizer=None).

:param gp: a given GP model
:param X: a pandas DataFrame of new training X, decoded
:param y: an array of new training y
:return: a GP model updated with the new training data.
"""
def update_model(gp, X, y, factors):
    y = np.array([y])
    #must encode matrix
    X_copy = encode_matrix(X, factors)
    X_copy_np = X_copy.to_numpy()

    #include old trained data from previous iterations
    all_X = np.vstack([gp.X_train_, X_copy_np])
    all_y = np.hstack([gp.y_train_, y])

    #get factor names
    names = [factor[0] for factor in factors.factors]

    #change back into pandas DataFrame
    all_X_df = pd.DataFrame(all_X,columns=names)
    
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
    #must encode matrix first
    X_encoded = encode_matrix(X,factors)

    gp.fit(X_encoded,y)
    return gp

"""
Use the given GP to predict a point, takes in a decoded Pandas DF.

:param gp: a given GP model
:param X: a pandas DataFrame of point in question, decoded.
:param factors: the factorset class instance
:return: an array with the distance prediction and standard deviation from the GP.
"""
def model_predict(gp,X,factors):
    #first encode X
    X_encoded = encode_matrix(X,factors)
    return gp.predict(X_encoded,return_std=True)