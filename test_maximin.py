from FactorSet import FactorSet
from InitStrategies import *

myFactorSet = FactorSet()

myFactorSet.add_continuous(name="draw_angle", minimum=0, maximum=180)
myFactorSet.add_ordinal(name="rubber_bands", levels=[1, 2, 3])
myFactorSet.add_categorical(name="projectile", levels=["pingpong", "whiffle"])

print("My factor set:", myFactorSet.factors)

(encoded_matrix,decoded_matrix) = maximin_design(myFactorSet,6,iters=5)

print("Encoded matrix: \n", encoded_matrix)
print("Decoded matrix: \n", decoded_matrix)
print("\n")
(encoded_matrix_2,decoded_matrix_2) = maximin_design(myFactorSet,6,iters=10000,plot_process=True)

print("Encoded matrix 2: \n", encoded_matrix_2)
print("Decoded matrix 2: \n", decoded_matrix_2)