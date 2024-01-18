from FactorSet import *
from InitStrategies import *
from gp_update_function import *
from PlotWrapper import *

"""
Creates a list of tuples of all possible combinations of discrete factors

:param factors: the FactorSet class corresponding to the design matrix
:return: a two lists of tuples containing all possible combinations of discrete factors at each level (one encoded, one decoded)
"""
def get_all_discrete_combinations_encoded(factors):
    #identify the discrete factors 
    #get a list of lists where each sub-list contains the "levels" in each discrete factor
    list_of_factor_ranges_decoded = []
    for factor in factors.factors:
        factor_range = factor[2]
        list_of_factor_ranges.append(factor_range)

    #



"""
Suggests a next experiment to conduct based on type_of_plan. 

:param X_design: pandas matrix containing experiments already executed, encoded
:param factors: the FactorSet class corresponding to the design matrix
:param gp: the current gp model
:param type_of_plan: string indicating whether to find experiments based on "Exploration" (uncertainty), "Exploitation" (lowest error value), "EI" (mix of both)
:param 
:return: an array containing the encoded next best experiment to execute
"""
def plan_next_experiment(X_design_encoded, factors, gp, type_of_plan):
    #create a list of tuples where all possible combos of discrete factors are enumerated

    #What if we have:
    #factors: [discrete, continuous, discrete]
    #want to preserve order in which the factors were added throughout the entire process
    
    #