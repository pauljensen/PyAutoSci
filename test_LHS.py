from FactorSet import FactorSet
from InitStrategies import scale_samples, latin_hypercube_design, encode_matrix, decode_matrix, generate_rand_vector_encoded

myFactorSet = FactorSet()

myFactorSet.add_continuous(name="draw_angle", minimum=0, maximum=180)
myFactorSet.add_ordinal(name="rubber_bands", levels=[1, 2, 3])
myFactorSet.add_categorical(name="projectile", levels=["pingpong", "whiffle"])

print(myFactorSet.factors)

(encoded_matrix,decoded_matrix) = latin_hypercube_design(myFactorSet, 6)
print("Encoded matrix: \n",encoded_matrix)
print("Decoded matrix: \n",decoded_matrix)
print("\n")

#test whether encode and decode matrix functions work
print("Encoded matrix after encoding:\n", encode_matrix(myFactorSet,decoded_matrix))
print("Decoded matrix after decoding:\n", decode_matrix(myFactorSet,encoded_matrix))

a = generate_rand_vector_encoded(myFactorSet)
print(a)