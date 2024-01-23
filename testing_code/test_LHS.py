from FactorSet import FactorSet
from InitStrategies import scale_samples, latin_hypercube_design, encode_matrix, decode_matrix, generate_rand_vector_encoded
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PlotWrapper import *
myFactorSet = FactorSet()

myFactorSet.add_continuous(name="draw_angle", minimum=0, maximum=180)
myFactorSet.add_ordinal(name="rubber_bands", levels=[1, 2, 3])
myFactorSet.add_categorical(name="projectile", levels=["pingpong", "whiffle"])

print(myFactorSet.factors)

decoded_matrix = latin_hypercube_design(myFactorSet, 6)
encoded_matrix = encode_matrix(decoded_matrix,myFactorSet)

print("Encoded matrix: \n",encoded_matrix)
print("Decoded matrix: \n",decoded_matrix)
print("\n")

#check if encode_matrix and decode_matrix work
print("Decocded matrix after using decoded_matrix function:\n",decode_matrix(encoded_matrix,myFactorSet))

plot_design(decoded_matrix,myFactorSet,"Maximin design decoded")