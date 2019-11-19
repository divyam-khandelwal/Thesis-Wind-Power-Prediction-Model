from random import choices
population = [1, 2, 3, 4, 5, 6]
weights = [0.999, 0.0499999, 0.05, 0.2, 0.4, 0.2]
print(choices(population, weights))
