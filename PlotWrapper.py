import plotly.express as px
from FactorSet import FactorSet
from mpl_toolkits.mplot3d import Axes3D
import copy

"""
Plots a given design (encoded or decoded) in a pandas DataFrame in 2D or 3D and displays it.

:param df: the pandas matrix
:param factors: an instance of the FactorSet class, ideally has factors filled out
:return: nothing
"""
def plot_design(X_design_df, factors, title="Plotted Design"): #encoded_or_decoded):

    # #if decoded, create a copy of the matrix with the categorical factors encoded
    # if encoded_or_decoded == "decoded":
    #     X_design_df_copy = X_design_df.copy()

    #     #go through each factor and find the ones that are "Categorical"
    #     for factor in factors.factors:

    #         #for every categorical factor, encode the column in X_design_df_copy
    #         if factor[2] == "Categorical":
    #             curr_name = factor[0]
    #             #turn the column into a category type
    #             category_column_series = X_design_df_copy[curr_name].astype("category")
    #             #now that the column is a category type, can encode it with ".cat.codes"
    #             X_design_df_copy[curr_name] = category_column_series.cat.codes
    
    # #if encoded, do not change X_design
    # elif encoded_or_decoded == "encoded":
    #     X_design_df_copy = X_design_df.copy()
    
    # #if neither of those, raise an error
    # else:
    #     raise ValueError("encoded_or_decoded param must be a string of either \'encoded\' or '\decoded\'.")

    #get number of factors
    num_factors = len(factors.factors)

    #get factors names, used for axes names for plot later
    names = [factor[0] for factor in factors.factors]

    #now that matrix is ready to plot, differentiate between plotting 2 vs 3 factors
    #if only plotting 2 factors, plot a 2D scatterplot
    if num_factors == 2:
        fig = px.scatter(X_design_df, 
                        x=name[0], 
                        y=name[1],
                        title=title)
    elif num_factors == 3:
        fig = px.scatter_3d(X_design_df, 
                            x=names[0],
                            y=names[1], 
                            z=names[2],
                            title=title)
    #if num_factors is not one of these, do not plot and raise an error
    else:
        raise ValueError("the number of factors must be 2 or 3.")
    
    fig.show()

    return


