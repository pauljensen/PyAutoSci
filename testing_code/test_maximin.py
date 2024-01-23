from FactorSet import FactorSet
from InitStrategies import *
from PlotWrapper import *

myFactorSet = FactorSet()

myFactorSet.add_continuous(name="draw_angle", minimum=0, maximum=180)
myFactorSet.add_ordinal(name="rubber_bands", levels=[1, 2, 3])
myFactorSet.add_categorical(name="projectile", levels=["pingpong", "whiffle"])

print("My factor set:", myFactorSet.factors)

decoded_matrix = maximin_design(myFactorSet,20,iters=5)
encoded_matrix = encode_matrix(decoded_matrix,myFactorSet)

print("Encoded matrix: \n", encoded_matrix)
print("Decoded matrix: \n", decoded_matrix)
print("\n")
plot_design(decoded_matrix,myFactorSet,"Maximin plot 5 iters")

decoded_matrix_2 = maximin_design(myFactorSet,20,iters=10000,plot_process=True)
encoded_matrix_2= encode_matrix(decoded_matrix_2,myFactorSet)
print("Encoded matrix 2: \n", encoded_matrix_2)
print("Decoded matrix 2: \n", decoded_matrix_2)

plot_design(decoded_matrix_2,myFactorSet,"Maximin plot 10000 iters")