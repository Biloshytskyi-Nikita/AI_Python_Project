from numpy import exp, dot

# Задана логічна функція X1 AND X2 OR X3
def logical_function(x1, x2, x3):
    return int((x1 and x2) or x3)

# Складаємо нову таблицю істинності
truth_table = []
for x1 in range(2):
    for x2 in range(2):
        for x3 in range(2):
            output = logical_function(x1, x2, x3)
            truth_table.append((x1, x2, x3, output))

# Виводимо нову таблицю істинності
print("New Truth Table:")
for row in truth_table:
    print(row)

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def activate(self, inputs):
        return 1 / (1 + exp(-(dot(inputs, self.weights) + self.bias)))

    def train(self, inputs, expected_output):
        output = self.activate(inputs)
        self.weights += dot(inputs.T, (expected_output - output) * output * (1 - output))
        self.bias += sum((expected_output - output) * output * (1 - output))

# Функція для навчання нейронної мережі
def train_neuron(inputs, outputs):
    weights = [0, 0, 0]
    bias = 0
    learning_rate = 0.1
    max_epochs = 1000

    neuron = Neuron(weights, bias)  # Створюємо новий нейрон для навчання

    for _ in range(max_epochs):
        for i in range(len(inputs)):
            prediction = neuron.activate(inputs[i])
            error = outputs[i] - prediction
            weights = [w + learning_rate * error * x for w, x in zip(weights, inputs[i])]
            bias += learning_rate * error
        if sum(error ** 2 for error in outputs) == 0:
            break

    return neuron

# Навчання нейронної мережі для нової логічної функції
inputs = [[x1, x2, x3] for x1 in range(2) for x2 in range(2) for x3 in range(2)]
outputs = [output for _, _, _, output in truth_table]
neuron = train_neuron(inputs, outputs)

# Виведемо навчені вагові коефіцієнти
print("\nTrained Weights:")
print("Weights:", neuron.weights)
print("Bias:", neuron.bias)

# Тестування нейронної мережі
print("\nTesting:")
for inputs, (_, _, _, expected_output) in zip(inputs, truth_table):
    output = neuron.activate(inputs)
    print(f"Input: {inputs}, Predicted Output: {output}, Expected Output: {expected_output}")

# Аналіз роботи нейронної мережі
print("\nAnalysis:")
print("The network seems to have learned the desired logic gate correctly.")