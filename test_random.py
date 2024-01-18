from FactorSet import FactorSet
from InitStrategies import *
from PlotWrapper import *
myFactorSet = FactorSet()

myFactorSet.add_continuous(name="draw_angle", minimum=0, maximum=180)
myFactorSet.add_ordinal(name="rubber_bands", levels=[1, 2, 3])
myFactorSet.add_categorical(name="projectile", levels=["pingpong", "whiffle"])

print("My factor set:", myFactorSet.factors)

decoded_matrix = random_design(myFactorSet,6)
encoded_matrix = encode_matrix(decoded_matrix,myFactorSet)
print("Encoded matrix:\n",encoded_matrix)
print("Decoded matrix:\n",decoded_matrix)

plot_design(decoded_matrix,myFactorSet,"Random design")