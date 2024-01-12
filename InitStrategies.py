from scipy.stats.qmc import LatinHypercube
import warnings
from FactorSet import FactorSet
import numpy as np
import pandas as pd
import math
import random
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

"""
Scale the LHS samples to the given bounds.

:param samples: LHS samples in [0, 1] range
:param bounds: List of tuples (min, max) for each dimension
:return: Scaled samples
"""
def scale_samples(samples, bounds):
    scaled_samples = []
    for col_idx in range(samples.shape[1]): 
        #for each column of samples, put the values in range according to bounds[col_idx]
        scaled_sample = samples[:,col_idx]*(bounds[col_idx][1]-bounds[col_idx][0])+bounds[col_idx][0]
        scaled_samples.append(scaled_sample)
    return np.array(scaled_samples).T


"""
Encode a given pandas matrix with discrete integers in place of the original ordinal or categorical values.
Also convert continuous variables to [0,1] scale. 

:param factors: an instance of the FactorSet class, ideally has factors filled out
:param df: the decoded pandas matrix
:return: encoded matrix
"""
def encode_matrix(factors, df):
    df_copy = df.copy()
    for factorIdx in range(len(factors.factors)):
        factor = factors.factors[factorIdx]
        name = factor[0]
        type_of_factor = factor[2]
        if type_of_factor == "Ordinal" or type_of_factor == "Categorical":
            #create a set mapping the levels to the range of integers
            levels = factor[1]
            mapping = dict()
            for level_idx in range(len(levels)):
                level = levels[level_idx]
                mapping[level] = float(level_idx)
            df_copy[name] = df_copy[name].replace(mapping)
        elif type_of_factor == "Continuous":
            #want to normalize this continuous factor into [0,1]
            factor_range = factor[1]
            minimum = factor_range[0]
            maximum = factor_range[1]

            #get the column from df_copy, and normalize everything in that column
            df_copy[name] = (df_copy[name]-minimum)/(maximum-minimum)
    return df_copy


"""
Decode a given pandas matrix (replace values with original ordinal and categorical factors).

:param factors: an instance of the FactorSet class, ideally has factors filled out
:param df: the encoded pandas matrix
:return: decoded matrix
"""
def decode_matrix(factors, df):
    df_copy = df.copy()
    for factorIdx in range(len(factors.factors)):
        factor = factors.factors[factorIdx]
        name = factor[0]
        type_of_factor = factor[2]
        if type_of_factor == "Ordinal" or type_of_factor == "Categorical":
            #create a set mapping the integers in the range to the levels
            levels = factor[1]
            mapping = dict()
            for idx in range(len(levels)):
                mapping[idx] = levels[idx]
            df_copy[name] = df_copy[name].replace(mapping)
        elif type_of_factor == "Continuous":
            #want to transform this factor from [0,1] to the original range
            factor_range = factor[1]
            minimum = factor_range[0]
            maximum = factor_range[1]

            df_copy[name] = df_copy[name] * (maximum-minimum) + minimum
    return df_copy


"""
Creates numbered ranges for ordinal and categorical factors. 

:param factors: an instance of the FactorSet class, ideally has factors filled out
:return: a list of ranges, based on the factors' min and max values or number of levels
"""
def create_modified_ranges(factors):
    modified_ranges = []
    for factor_idx in range(len(factors.factors)):
        factor = factors.factors[factor_idx]
        if factor[2] == "Categorical" or factor[2] == "Ordinal":
            num_levels = len(factor[1])
            modified_ranges.append([0,num_levels-1])
        
        elif factor[2] == "Continuous":
            modified_ranges.append(factor[1])
    return modified_ranges


"""
Create a Latin Hypercube design. Uses scipy's LatinHypercube function.

:param factors: an instance of the FactorSet class, ideally has factors filled out
:param n: the number of samples with which to populate the LHS design
:return: a pandas dataframe where each row is a point in the LHS design
"""
def latin_hypercube_design(factors, n):

    #check whether there are both ordinal/categorical and continuous factors
    #if there are ordinal/categorical factors, create a separate list of ranges
    #also keep track of indices of discrete factors
    discrete_present_bool = False
    continuous_present_bool = False

    for factor_idx in range(len(factors.factors)):
        factor = factors.factors[factor_idx]
        if factor[2] == "Categorical" or factor[2] == "Ordinal":
            discrete_present_bool = True
        elif factor[2] == "Continuous":
            continuous_present_bool = True
    
    if discrete_present_bool and continuous_present_bool:
        warnings.warn("Using both discrete and continuous factors will not result in a true LHS design. The design will be made as if the discrete was continuous, then rounded afterwards.", UserWarning)

    num_dimensions = len(factors.factors)
    #call the library LHS function with the number of factors as the dimensions
    LHS_object = LatinHypercube(num_dimensions)
    LHS_samples_unrounded = LHS_object.random(n)

    #create a pandas dataframe out of this with column names
    col_names = [factor[0] for factor in factors.factors]
    cont_encoded_discrete_unrounded_matrix_df = pd.DataFrame(LHS_samples_unrounded,columns=col_names)

    #want to round each value in the discrete columns to the nearest integers
    encoded_matrix_df = cont_encoded_discrete_unrounded_matrix_df.copy()
    for factor_idx in range(len(factors.factors)):
        factor = factors.factors[factor_idx]
        name = factor[0]
        if factor[2] == "Ordinal" or factor[2] == "Categorical":
            encoded_matrix_df[name] = encoded_matrix_df[name].round()

    #must decode the df_LHS_samples_encoded_copy
    #encoded_matrix_df_copy = encoded_matrix_df.copy()
    decoded_matrix_df = decode_matrix(factors, encoded_matrix_df)
    
    #return both decoded and encoded Pandas dataframes
    return (encoded_matrix_df,decoded_matrix_df)


"""
Calculate Euclidean distance between two vectors

:param vector1_np_array: the first vector in a numpy array
:param vector2_np_array: the second vector in a numpy array
:return: the Euclidean distance between vectors 1 and 2
"""
def euclidean_distance(vector1_np_array, vector2_np_array):
    summation = 0
    for idx in range(len(vector1_np_array)):
        elem1 = vector1_np_array[idx]
        elem2 = vector2_np_array[idx]
        summation += (elem1-elem2)**2
    return math.sqrt(summation)


"""
Calculate average pairwise Euclidean distance of a numpy matrix

:param matrix: the numpy matrix, calculating distances between rows
:return: the average pairwise Euclidean distance
"""
def calc_avg_pairwise_euc_dist(matrix):
    summation = 0
    pairwise_distances = pdist(matrix, 'euclidean')
    return np.mean(pairwise_distances)


"""
Randomly generate a vector of factors given a FactorSet, decoded

:param factors: an instance of the FactorSet class, ideally has factors filled out
:return: a list of lists (vector) of randomly generated factors
"""
def generate_rand_vector_decoded(factors):
    vector = []
    num_factors = len(factors.factors)

    #loop through every factor
    for factor_idx in range(num_factors):
        #get current factor
        curr_factor = factors.factors[factor_idx]
        name = curr_factor[0]
        range_or_levels = curr_factor[1]
        factor_type = curr_factor[2]

        #if factor is continuous, randomly draw between min and max
        if factor_type == "Continuous":
            min_val = range_or_levels[0]
            max_val = range_or_levels[1]
            rand_value = random.random()
            rand_value_scaled = rand_value * (max_val-min_val) + min_val
            vector.append(rand_value_scaled)
        #if factor is ordinal or categorical, randomly draw from the levels
        elif factor_type == "Ordinal" or factor_type == "Categorical":
            rand_value = random.choice(range_or_levels)
            vector.append(rand_value)
    return vector


"""
Randomly generate a vector of factors given a FactorSet, encoded
Discrete variables are [0, ..., number of levels], continuous values are from [0,1] instead of original ranges

:param factors: an instance of the FactorSet class, ideally has factors filled out
:return: a list of lists (vector) of randomly generated factors
"""
def generate_rand_vector_encoded(factors):
    vector = []
    num_factors = len(factors.factors)

    #loop through every factor
    for factor_idx in range(num_factors):
        #get current factor
        curr_factor = factors.factors[factor_idx]
        name = curr_factor[0]
        range_or_levels = curr_factor[1]
        factor_type = curr_factor[2]

        #if factor is continuous, randomly draw between min and max
        if factor_type == "Continuous":
            min_val = range_or_levels[0]
            max_val = range_or_levels[1]
            rand_value = random.random()
            vector.append(rand_value)
        #if factor is ordinal or categorical, randomly draw from the levels
        elif factor_type == "Ordinal" or factor_type == "Categorical":
            #create mapping between levels and integers
            rand_value = random.choice(range_or_levels)
            #print(rand_value)
            encoded_rand_value = float(range_or_levels.index(rand_value))
            vector.append(encoded_rand_value)
    return vector


"""
Create a maximin design

:param factors: an instance of the FactorSet class, ideally has factors filled out
:param n: the number of samples with which to populate the maximin design
:param iters: the number of iterations the exchange algorithm will execute
:param plot_process: whether or not to plot the average pairwise Euc distances over the iterations
:return: a pandas dataframe where each row is a point in the maximin design
"""
def maxmimin_design(factors,n,iters=100,plot_process=False):

    #for keeping track of the average pairwise Euclidean distances for plotting
    all_avg_pairwise_euc_dists = []

    #the collection of points used for the maximin design
    encoded_vectors = []

    #create a list of randomly generated vectors
    for vector_idx in range(n):
        rand_vector = generate_rand_vector_encoded(factors)
        encoded_vectors.append(rand_vector)

    #turn encoded_matrix_df into encoded_matrix_np for Euclidean dist calculations
    encoded_matrix_np = np.array(encoded_vectors)
    
    #for each iteration
    for iter_num in range(iters):
        #1) randomly generate an vector
        potential_replacement_encoded = generate_rand_vector_encoded(factors)
        
        #2) randomly select a vector to potentially replace in encoded_matrix_np
        rand_idx = random.choice(list(range(n)))

        #3) evaluate if the replacement will improve the average pairwise Euclidean distance
        encoded_matrix_np_copy = encoded_matrix_np.copy()
        encoded_matrix_np_copy[rand_idx] = potential_replacement_encoded

        potential_avg_pairwise_euc_dist = calc_avg_pairwise_euc_dist(encoded_matrix_np_copy)
        avg_pairwise_avg_euc_dist = calc_avg_pairwise_euc_dist(encoded_matrix_np)
        #if yes, do the swap
        if potential_avg_pairwise_euc_dist > avg_pairwise_avg_euc_dist:
            encoded_matrix_np = encoded_matrix_np_copy
            all_avg_pairwise_euc_dists.append(potential_avg_pairwise_euc_dist)
        #if not, do not do the swap, just record the avg pairwise euc dist
        else:
            all_avg_pairwise_euc_dists.append(avg_pairwise_avg_euc_dist)
    
    #turn encoded_matrix_np into a dataframe
    col_names = [factor[0] for factor in factors.factors]
    encoded_matrix_df = pd.DataFrame(encoded_matrix_np,columns=col_names)
    #encoded_matrix_df_copy = encoded_matrix_df.copy()
    #decode the encoded matrix
    decoded_matrix_df = decode_matrix(factors,encoded_matrix_df)

    #if plot is true, generate a plot of all_avg_pairwise_euc_dists and display it
    if plot_process:
        plt.plot(all_avg_pairwise_euc_dists)
        plt.title("Average pairwise distance over time")
        plt.xlabel("Iteration")
        plt.ylabel("Average pairwise distance")
        plt.show()

    #return the encoded and decoded dfs
    return (encoded_matrix_df, decoded_matrix_df)


"""
Create a random design

:param factors: an instance of the FactorSet class, ideally has factors filled out
:param n: the number of samples with which to populate the random design
:return: a pandas dataframe where each row is a point in the random design
"""
def random_design(factors,n):
    #loop over range of n and generate random encoded vector for each one
    rand_vectors_encoded = []
    for point in range(n):
        rand_vector = generate_rand_vector_encoded(factors)
        rand_vectors_encoded.append(rand_vector)
    
    #turn rand_vectors into a pandas dataframe
    col_names = [factor[0] for factor in factors.factors]
    rand_vectors_encoded_df = pd.DataFrame(rand_vectors_encoded,columns=col_names)
    #rand_vectors_encoded_df_copy = rand_vectors_encoded_df.copy()
    
    #decode the rand_vectors_encoded_df_copy
    rand_vectors_decoded_df = decode_matrix(factors,rand_vectors_encoded_df)

    #return both encoded and decoded pd frames
    return (rand_vectors_encoded_df,rand_vectors_decoded_df)
