from FactorSet import *
from InitStrategies import *
from gp_update_function import *
from PlotWrapper import *
import itertools
from scipy.stats import norm
from scipy.optimize import minimize

    
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

    return random_sampled_unscaled


"""
Suggests a next experiment to conduct based on policy. 

:param gp: the current gp model
:param factors: the FactorSet class corresponding to the design matrix
:param policy: string indicating whether to find experiments based on "Exploration" (uncertainty), "Exploitation" (lowest error value), "EI" (mix of both)
:return: an array containing the encoded next best experiment to execute and the resulting objective value
"""
def plan_next_experiment(gp, factors, policy, random_restarts = 100):

    #retrieve the levels lists of the discrete factors
    levels_list = []
    for factor in factors.factors:
        type_of_factor = factor[2]
        if type_of_factor == "Ordinal" or type_of_factor == "Categorical":
            #need to encode these factors, generate a mapping
            #each level corresponds to an integer
            #e.g. ["ping pong", "whiffle"] => [0,1]
            #e.g. [1,2,3] => [0,1,2]
            len_levels = len(factor[1])
            encoded_levels = list(range(len_levels))
            levels_list.append(encoded_levels)

    #create a list of tuples where all possible combos of discrete factors are enumerated
    #go in the order of the discrete factors (the last factors)
    #continuous factors will always come first before the discrete factors
    all_discrete_poss_iterobj = itertools.product(*levels_list)
    all_discrete_poss = list(all_discrete_poss_iterobj)

    #keep track of all possible points to go for and take the one that improves objective the most
    all_possible_improvements_X_continuous = []
    all_possible_improvements_X_discrete = []
    all_possible_improvements_y = []

    #names of factors
    factor_names = [factor[0] for factor in factors.factors]

    #retrieve the bounds for the continuous factors, just [0,1]
    bounds = []
    for factor in factors.factors:
        if factor[2] == "Continuous":
            bounds.append([0,1])

    #go through the number of random restarts
    for rr in range(random_restarts):

        #define random starting point
        x0 = generate_random_continuous(factors)

        #for each possible discrete combination (in our case, 6):
        for discrete_combo_non_list in all_discrete_poss:
            discrete_combo = np.array(list(discrete_combo_non_list))
            all_possible_improvements_X_discrete.append(discrete_combo)

            #define the exploration, exploitation, and EI objective functions
            #exploration objective returns the uncertainty from the GP
            def exploration(continuous_input):
                x_numpy = np.hstack((continuous_input,discrete_combo))
                x_dataframe = pd.DataFrame([x_numpy],columns=factor_names)
                mu, sigma = gp.predict(x_dataframe, return_std=True)
                return -sigma[0]

            #exploitation returns the prediction (the catapult distance error) from the GP
            def exploitation(continuous_input):
                x_numpy = np.hstack((continuous_input,discrete_combo))
                x_dataframe = pd.DataFrame([x_numpy],columns=factor_names)
                mu, sigma = gp.predict(x_dataframe, return_std=True)
                return mu

            #EI returns a mixture of exploration and exploitation
            def EI(potential_x, gp):
                y_min = np.min(gp.y_train_)
                mu, sigma = gp.predict(potential_x, return_std=True)

                with np.errstate(divide='warn'):
                    all_zeros = np.zeros(np.size(mu))
                    improvement = np.maximum(all_zeros,y_min - mu)
                    Z = improvement / sigma
                    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
                    ei[sigma == 0.0] = 0.0

                return ei
            
            #the objective function to give the minimize call, inputting y_min_responses found earlier and the gp
            def min_ei(continuous_input):
                x_numpy = np.hstack((continuous_input,discrete_combo))
                x_dataframe = pd.DataFrame([x_numpy],columns=factor_names)
                return -EI(x_dataframe, gp).ravel()

            #define res at first
            res = None

            #want to perform an optimization
            #if doing exploration
            if policy == "Exploration":
                #do optimization with exploration objective function, want to maximize uncertainty
                res = minimize(exploration,x0,bounds=bounds,method='L-BFGS-B')

            #if doing exploitation
            elif policy == "Exploitation":
                #do optimization with exploitation objective function, want to minimize distance
                res = minimize(exploitation,x0,bounds=bounds,method='L-BFGS-B')
            
            #if doing EI
            elif policy == "EI":
                #do optimization with EI objective function, want to maximize exepcted improvement
                res = minimize(min_ei,x0,bounds=bounds,method='L-BFGS-B')
            
            #if the policy does not match any of the strings above, return an error
            else:
                raise ValueError("The policy must be Exploration, Exploitation or EI.")
            
            #add the res.x and res.fun to the all_possible_improvements arrays
            all_possible_improvements_X_continuous.append(res.x)
            all_possible_improvements_y.append(res.fun)
    
    #next, go through all collected optima and find the aboslute best one
    next_suggested_experiment = None
    obj_val = None
        
    #if doing EI or exploration, we are maximizing so find index of max val within all_possible_improvements_y
    if policy == "EI" or policy == "Exploration":
        #multiply all_possible_improvements_y by -1 because these objective values are negative
        all_possible_improvements_y_true = [(elem*-1) for elem in all_possible_improvements_y]
        obj_val = max(all_possible_improvements_y_true)
        max_index = all_possible_improvements_y_true.index(obj_val)

        next_suggested_experiment_continuous = all_possible_improvements_X_continuous[max_index]
        next_suggested_experiment_discrete = all_possible_improvements_X_discrete[max_index]
    
    #if doing exploitation, we are minimizing distance error, so find index of min val within all_possible_improvements_y
    elif policy == "Exploitation":
        obj_val = min(all_possible_improvements_y)
        min_index = all_possible_improvements_y.index(obj_val)

        next_suggested_experiment_continuous = all_possible_improvements_X_continuous[min_index]
        next_suggested_experiment_discrete = all_possible_improvements_X_discrete[min_index]

    #append the continuous and discrete together, then create a pandas df to return
    next_suggested_experiment = np.hstack((next_suggested_experiment_continuous,next_suggested_experiment_discrete))
    next_suggested_experiment_df_encoded = pd.DataFrame([next_suggested_experiment],columns=factor_names)

    #need to decode the entire next suggested experiment
    next_suggested_experiment_df_decoded = decode_matrix(next_suggested_experiment_df_encoded,factors)
    
    return next_suggested_experiment_df_decoded, obj_val

    