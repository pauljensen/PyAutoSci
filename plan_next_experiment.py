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
    #random_sampled_scaled = random_sampled_unscaled.copy()

    # #scale the samples according to the factors ranges
    # for factor_idx in range(len(factors.factors)):
    #     factor = factors.factors[factor_idx]
    #     if factor[2] == "Continuous":
    #         factor_range = factor[1]
    #         minimum = factor_range[0]
    #         maximum = factor_range[1]
    #         random_sampled_scaled[factor_idx] = random_sampled_unscaled[factor_idx] * (maximum-minimum) + minimum
    
    return random_sampled_unscaled


"""
Suggests a next experiment to conduct based on type_of_plan. 

:param y_responses: a numpy array of the responses from the catapult so far
:param factors: the FactorSet class corresponding to the design matrix
:param gp: the current gp model
:param type_of_plan: string indicating whether to find experiments based on "Exploration" (uncertainty), "Exploitation" (lowest error value), "EI" (mix of both)
:return: an array containing the encoded next best experiment to execute and the resulting objective value
"""
def plan_next_experiment(gp, factors, type_of_plan, random_restarts = 100):

    #collect y responses
    y_responses = gp.y_train_
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

        #if rr%100 == 0:
        #    print("On random restart number ",rr)

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
                x_numpy = np.hstack((continuous_input,discrete_combo))
                x_dataframe = pd.DataFrame([x_numpy],columns=factor_names)
                return -EI(x_dataframe, y_min_responses, gp).ravel()

            #define res at first
            res = None

            #want to perform an optimization
            #if doing exploration
            if type_of_plan == "Exploration":
                #do optimization with exploration objective function, want to maximize uncertainty
                res = minimize(exploration,x0,bounds=bounds,method='L-BFGS-B')

            #if doing exploitation
            elif type_of_plan == "Exploitation":
                #do optimization with exploitation objective function, want to minimize distance
                res = minimize(exploitation,x0,bounds=bounds,method='L-BFGS-B')
            
            #if doing EI
            elif type_of_plan == "EI":
                #do optimization with EI objective function, want to maximize exepcted improvement
                res = minimize(min_ei,x0,bounds=bounds,method='L-BFGS-B')
            
            #if the type_of_plan does not match any of the strings above, return an error
            else:
                raise ValueError("The type_of_plan must be Exploration, Exploitation or EI.")
            
            #add the res.x and res.fun to the all_possible_improvements arrays
            all_possible_improvements_X_continuous.append(res.x)
            all_possible_improvements_y.append(res.fun)
    
    #next, go through all collected optima and find the aboslute best one
    next_suggested_experiment = None
    obj_val = None

    # print("all_possible_improvements_X_continuous\n",all_possible_improvements_X_continuous)
    # print("\n")
    # print("all_possible_improvements_X_discrete\n",all_possible_improvements_X_discrete)
    # print("\n")
    # print("all_possible_improvements_y\n",all_possible_improvements_y)
    # print("\n")
        
    #if doing EI or exploration, we are maximizing so find index of max val within all_possible_improvements_y
    if type_of_plan == "EI" or type_of_plan == "Exploration":
        #multiply all_possible_improvements_y by -1 because these objective values are negative
        all_possible_improvements_y_true = [(elem*-1) for elem in all_possible_improvements_y]
        obj_val = max(all_possible_improvements_y_true)
        max_index = all_possible_improvements_y_true.index(obj_val)

        next_suggested_experiment_continuous = all_possible_improvements_X_continuous[max_index]
        next_suggested_experiment_discrete = all_possible_improvements_X_discrete[max_index]
    
    #if doing exploitation, we are minimizing distance error, so find index of min val within all_possible_improvements_y
    elif type_of_plan == "Exploitation":
        obj_val = min(all_possible_improvements_y)
        min_index = all_possible_improvements_y.index(obj_val)

        next_suggested_experiment_continuous = all_possible_improvements_X_continuous[min_index]
        next_suggested_experiment_discrete = all_possible_improvements_X_discrete[min_index]

    #append the continuous and discrete together, then create a pandas df to return
    next_suggested_experiment = np.hstack((next_suggested_experiment_continuous,next_suggested_experiment_discrete))
    next_suggested_experiment_df_encoded = pd.DataFrame([next_suggested_experiment],columns=factor_names)
    #print("next_suggested_experiment_df_encoded\n",next_suggested_experiment_df_encoded)
    #print("\n")

    #need to convert discrete factors in next_suggested_experiment_df_discrete_encoded back into decoded terms
    # next_suggested_experiment_df = next_suggested_experiment_df_discrete_encoded.copy()
    # for factor in factors.factors:
    #     if factor[2] == "Ordinal" or factor[2] == "Categorical":
    #         #create mapping between levels
    #         levels = factor[1]
    #         name = factor[0]
    #         mapping = dict()
    #         for idx in range(len(levels)):
    #             mapping[idx] = levels[idx]
    #         next_suggested_experiment_df[name] = next_suggested_experiment_df[name].replace(mapping)

    #need to decode the entire next suggested experiment
    next_suggested_experiment_df_decoded = decode_matrix(next_suggested_experiment_df_encoded,factors)
    
    return next_suggested_experiment_df_decoded, obj_val

    