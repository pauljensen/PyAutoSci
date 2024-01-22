import matplotlib.pyplot as plt
from FactorSet import *
from gp_update_function import *
from InitStrategies import *
from PlotWrapper import *
from plan_next_experiment import *
import plotly.express as px

    
"""
Randomly generate an array of continuous factors from the given factor set.

:param gp: the Gaussian Process model whose responses need to be plotted
:param factors: the FactorSet class instance
:param X: the experiments performed so far
:param y: the results of the experiments performed so far
:return: nothing
"""
def plot_heatmaps(gp,factors):

    #get column names for dataframes
    factor_names = [factor[0] for factor in factors.factors]

    #get all the ranges
    ranges = dict()
    for factor in factors.factors: 
        name = factor[0]
        range_or_level = factor[1]
        ranges[name] = range_or_level
    
    #create draw_angle range and rubber_bands range
    min_draw_angle_encoded = 0
    max_draw_angle_encoded = 1
    draw_angle_range = np.linspace(min_draw_angle_encoded,max_draw_angle_encoded,1000)

    #create rubber_bands range
    num_rubber_bands = len(ranges["rubber_bands"])
    min_rubber_bands = 0
    max_rubber_bands = num_rubber_bands-1
    rubber_band_range = np.linspace(min_rubber_bands,max_rubber_bands,1000)

    #create meshgrid for plotting (flatten later)
    draw_angle_grid, rubber_band_grid = np.meshgrid(draw_angle_range,rubber_band_range)
    #print("draw_angle_grid:",draw_angle_grid)
    #print("rubber_band_grid:",rubber_band_grid)

    #create encoded list for number of projectiles
    num_projectiles = len(ranges["projectile"])
    projectile_range = list(range(num_projectiles))

    #retrieve decoded minimum and maximum for draw_angle and rubber_bands
    for factor in factors.factors:
        name = factor[0]
        if name == "draw_angle":
            draw_angle_min_decoded = factor[1][0]
            draw_angle_max_decoded = factor[1][1]
        elif name == "rubber_bands":
            rubber_bands_min_decoded = factor[1][0]
            rubber_bands_max_decoded = factor[1][-1]
        
    #plot predictions and standard deviations
    for projectile_value in projectile_range:
        projectile_value_levels = ranges["projectile"]
        projectile_value_decoded = projectile_value_levels[projectile_value]
        
        #print("draw_angle_grid ravel:",draw_angle_grid.ravel())
        
        # Create input array for predictions: combine draw_angle_grid, rubber_band_grid with fixed projectile_value
        list_to_turn_to_Xpred = []
        for factor in factors.factors:
            name = factor[0]
            if name == "rubber_bands":
                list_to_turn_to_Xpred.append(rubber_band_grid.ravel())
            elif name == "projectile":
                list_to_turn_to_Xpred.append(np.full(draw_angle_grid.size, projectile_value))
            elif name == "draw_angle":
                list_to_turn_to_Xpred.append(draw_angle_grid.ravel())

        #assemble dataframe of X_predictions
        X_pred = np.array(list_to_turn_to_Xpred).T

        X_pred = pd.DataFrame(X_pred,columns=factor_names)

        # Predict using the GP
        y_pred, y_std = gp.predict(X_pred, return_std=True)  

        #y_pred = np.absolute(y_pred)

        # Reshape for plotting
        y_pred = y_pred.reshape(draw_angle_grid.shape)

        y_std = y_std.reshape(draw_angle_grid.shape)

        # Plotting the heatmap
        plt.figure()
        plt.imshow(y_pred, extent=[draw_angle_min_decoded, draw_angle_max_decoded, rubber_bands_min_decoded, rubber_bands_max_decoded],
                origin='lower', aspect='auto',cmap='hot')
        plt.colorbar(label='GP Response')
        plt.xlabel('Draw angle')
        plt.ylabel('Number of rubber bands')
        plt.title('Distance Heatmap for '+projectile_value_decoded)
        plt.show()

        # Plotting the heatmap for GP standard deviation
        plt.figure()
        plt.imshow(y_std, extent=[draw_angle_min_decoded, draw_angle_max_decoded, rubber_bands_min_decoded, rubber_bands_max_decoded],
                origin='lower', aspect='auto',cmap='hot')
        plt.colorbar(label='GP Standard Deviation')
        plt.xlabel('Draw angle')
        plt.ylabel('Number of rubber bands')
        plt.title('GP Standard Deviation Heatmap for '+projectile_value_decoded)
        plt.show()

        #need to make a plot for EI
