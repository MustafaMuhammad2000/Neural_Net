import NeuralNet as MN
import numpy as np

# Using the neural network!
test_neural_network = MN.neural_network([2, 3, 3, 1], 0.2)


# Testing a simple XOR bitwise function
input_arr = [[0, 0], [0, 1], [1, 0], [1, 1]]
target_arr = [[0], [1], [1], [0]]


print("Pre-training results")
print(test_neural_network.test(input_arr[0]))
print(test_neural_network.test(input_arr[1]))
print(test_neural_network.test(input_arr[2]))
print(test_neural_network.test(input_arr[3]))

for i in range(100000):
    rand = np.random.randint(0, 4)
    test_neural_network.train(input_arr[rand], target_arr[rand])

print("Training complete, now time for post-training results")
print(test_neural_network.test(input_arr[0]))
print(test_neural_network.test(input_arr[1]))
print(test_neural_network.test(input_arr[2]))
print(test_neural_network.test(input_arr[3]))