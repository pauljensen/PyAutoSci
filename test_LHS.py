from FactorSet import FactorSet
from InitStrategies import scale_samples, latin_hypercube_design, encode_matrix, decode_matrix, generate_rand_vector_encoded
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
myFactorSet = FactorSet()

myFactorSet.add_continuous(name="draw_angle", minimum=0, maximum=180)
myFactorSet.add_ordinal(name="rubber_bands", levels=[1, 2, 3])
myFactorSet.add_categorical(name="projectile", levels=["pingpong", "whiffle"])

print(myFactorSet.factors)

(encoded_matrix,decoded_matrix) = latin_hypercube_design(myFactorSet, 6)
print("Encoded matrix: \n",encoded_matrix)
print("Decoded matrix: \n",decoded_matrix)
print("\n")

#check if encode_matrix and decode_matrix work
print("Encoded matrix after using encode_matrix function:\n",encode_matrix(myFactorSet,decoded_matrix))
print("Encoded matrix after using encode_matrix function again:\n",encode_matrix(myFactorSet,decoded_matrix))
print("Decocded matrix after using decoded_matrix function:\n",decode_matrix(myFactorSet,encoded_matrix))
print("Decocded matrix after using decoded_matrix function again:\n",decode_matrix(myFactorSet,encoded_matrix))