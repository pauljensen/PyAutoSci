from FactorSet import *
from InitStrategies import *
from gp_update_function import *
from PlotWrapper import *
import itertools
from scipy.stats import norm

"""
Randomly generate an array of continuous factors from the given factor set.

:param factors: the FactorSet class instance
:return: an array containing the randomly generated continuous factors, decoded.
"""
def generate_random_continuous(factors):
    #get number of continuous factors
    num_continuous = 0
    for factor in factors.factors:
        if factor[2] == "Continuous":
            num_continuous += 1
    
    #create randomly sampled array of number of continuous factors
    random_sampled_unscaled = np.random.uniform(size=num_continuous)
    random_sampled_scaled = random_sampled_unscaled.copy()

    #scale the samples according to the factors ranges
    for factor_idx in range(len(factors.factors)):
        factor = factors.factors[factor_idx]
        factor_range = factor[1]
        minimum = factor_range[0]
        maximum = factor_range[1]
        random_sampled_scaled[factor_idx] = random_sampled_unscaled[factor_idx] * (maximum-minimum) + minimum
    
    return random_sampled_scaled


"""
Suggests a next experiment to conduct based on type_of_plan. 

:param X_design: pandas matrix containing experiments already executed, encoded
:param y_responses: a numpy array of the responses from the catapult so far
:param factors: the FactorSet class corresponding to the design matrix
:param gp: the current gp model
:param type_of_plan: string indicating whether to find experiments based on "Exploration" (uncertainty), "Exploitation" (lowest error value), "EI" (mix of both)
:return: an array containing the encoded next best experiment to execute and the resulting objective value
"""
def plan_next_experiment(X_design_encoded, y_responses, factors, gp, type_of_plan, random_restarts = 1000):

    #retrieve the levels lists of the discrete factors
    levels_list = []
    for factor in factors.factors:
        type_of_factor = factor[2]
        if type_of_factor == "Ordinal" or type_of_factor == "Categorical":
            levels_list.append(factor[1])

    #create a list of tuples where all possible combos of discrete factors are enumerated
    #go in the order of the discrete factors (the last factors)
    #continuous factors will always come first before the discrete factors
    all_discrete_poss_iterobj = itertools.product(*levels_list)
    all_discrete_poss = list(all_discrete_poss_iterobj)

    #keep track of all possible points to go for and take the one that improves objective the most
    all_possible_improvements_X = []
    all_possible_improvements_y = []

    #retrieve the bounds for the continuous factors
    bounds = []
    for factor in factors.factors:
        if factor[2] == "Continuous":
            bounds.append(factor[1])

    #go through the number of random restarts
    for rr in range(random_restarts):
        #define random starting point
        x0 = generate_random_continuous(factors)

        #for each possible discrete combination (in our case, 6):
        for discrete_combo in all_discrete_poss:

            #define the exploration, exploitation, and EI objective functions
            #exploration objective returns the uncertainty from the GP
            def exploration(continuous_input):
                x = np.array([continuous_input+discrete_combo])
                mu, sigma = gp.predict(x, return_std=True)
                return -sigma[0]

            #exploitation returns the prediction (the catapult distance error) from the GP
            def exploitation(continuous_input):
                x = np.array([continuous_input+discrete_combo])
                mu, sigma = gp.predict(x, return_std=True)
                return mu

            #EI returns a mixture of exploration and exploitation
            def EI(potential_x, y_min, gp):
                mu, sigma = gp.predict(potential_x, return_std=True)

                with np.errstate(divide='warn'):
                    improvement = y_min - mu
                    Z = improvement / sigma
                    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
                    ei[sigma == 0.0] = 0.0

                return ei
            
            #get the minimum y from y_responses
            y_min_responses = np.min(y_responses)
            
            #the objective function to give the minimize call, inputting y_min_responses found earlier and the gp
            def min_ei(continuous_input):
                x = np.array([continuous_input+discrete_combo])
                return -EI(x, y_min_responses, gp).ravel()

            #define res at first
            res = None

            #want to perform an optimization
            #if doing exploration
            if type_of_plan == "Exploration":
                #do optimization with exploration objective function
                res = minimize(exploration,x0,bound=bounds,method='L-BFGS-B')

            #if doing exploitation
            elif type_of_plan == "Exploitation":
                #do optimization with exploitation objective function
                res = minimize(exploitation,x0,bounds=bounds,method='L-BFGS-B')
            
            #if doing EI
            elif type_of_plan == "EI":
                #do optimization with EI objective function
                res = minimize(min_ei,x0,bounds=bounds,method='L-BFGS-B')
            
            #if the type_of_plan does not match any of the strings above, return an error
            else:
                raise ValueError("The type_of_plan must be Exploration, Exploitation or EI.")
            
            #add the res.x and res.fun to the all_possible_improvements arrays
            all_possible_improvements_X.append(res.x)
            all_possible_improvements_y.append(res.fun)
    
    #next, go through all collected optima and find the aboslute best one
    next_suggested_experiment = None
    obj_val = None
        
    #if doing EI or exploration, we are maximizing so find index of max val within all_possible_improvements_y
    if type_of_plan == "EI" or type_of_plan == "Exploration":
        obj_val = max(all_possible_improvements_y)
        max_index = all_possible_improvements_y.index(obj_val)

        next_suggested_experiment = all_possible_improvements_X[max_index]
    
    #if doing exploitation, we are minimizing distance error, so find index of min val within all_possible_improvements_y
    elif type_of_plan == "Exploitation":
        obj_val = min(all_possible_improvements_y)
        min_index = all_possible_improvements_y.index(obj_val)

        next_suggested_experiment = all_possible_improvements_X[min_index]
    
    return next_suggested_experiment, obj_val

    