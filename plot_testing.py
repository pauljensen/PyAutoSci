from FactorSet import FactorSet
from InitStrategies import *
import pandas as pd
import plotly.express as px
from PlotWrapper import *

myFactorSet = FactorSet()

myFactorSet.add_continuous(name="draw_angle", minimum=0, maximum=180)
myFactorSet.add_ordinal(name="rubber_bands", levels=[1, 2, 3])
myFactorSet.add_categorical(name="projectile", levels=["pingpong", "whiffle"])

print("My factor set:", myFactorSet.factors)

#maximin design plotting
decoded_matrix = maximin_design(myFactorSet,20,iters=10000,plot_process=True)
encoded_matrix = encode_matrix(decoded_matrix,myFactorSet)
print("Encoded matrix maximin: \n", encoded_matrix)
print("Decoded matrix maximin: \n", decoded_matrix)

print("Plotting the encoded design maximin:\n")
plot_design(encoded_matrix,myFactorSet,"Maximin design encoded")

print("plotting the decoded design maximin:\n")
plot_design(decoded_matrix,myFactorSet,"Maximin design decoded")

#LHS design plotting
decoded_matrix_LHS = latin_hypercube_design(myFactorSet, 20)
encoded_matrix_LHS = encode_matrix(decoded_matrix_LHS, myFactorSet)

print("Encoded matrix LHS: \n", encoded_matrix_LHS)
print("Decoded matrix LHS: \n", decoded_matrix_LHS)

print("Plotting the encoded design LHS:\n")
plot_design(encoded_matrix_LHS,myFactorSet,"LHS design encoded")

print("plotting the decoded design LHS:\n")
plot_design(decoded_matrix_LHS,myFactorSet,"LHS design decoded")

#Random design plotting
decoded_matrix_random = random_design(myFactorSet,20)
encoded_matrix_random = encode_matrix(decoded_matrix_random,myFactorSet)

print("Encoded matrix random: \n", encoded_matrix_random)
print("Decoded matrix random: \n", decoded_matrix_random)

print("Plotting the encoded design LHS:\n")
plot_design(encoded_matrix_random,myFactorSet,"Random design encoded")

print("plotting the decoded design LHS:\n")
plot_design(decoded_matrix_random,myFactorSet,"Random design decoded")

