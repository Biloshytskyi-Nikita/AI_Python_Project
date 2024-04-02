import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Завантаження даних про квіти Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Розбиття даних на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормалізація даних
X_train_normalized = (X_train - np.min(X_train)) / (np.max(X_train) - np.min(X_train))
X_test_normalized = (X_test - np.min(X_train)) / (np.max(X_train) - np.min(X_train))

# Визначення архітектури нейронної мережі
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Ініціалізація ваг та зміщень
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    # Сигмоїдна функція активації
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Пряме поширення
    def forward(self, X):
        self.hidden_sum = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.activated_hidden = self.sigmoid(self.hidden_sum)

        self.output_sum = np.dot(self.activated_hidden, self.weights_hidden_output) + self.bias_output
        self.activated_output = self.sigmoid(self.output_sum)

        return self.activated_output

    # Зворотне поширення
    def backward(self, X, y, output, learning_rate=0.01):
        self.output_error = y - output
        self.output_delta = self.output_error * (output * (1 - output))

        self.hidden_error = np.dot(self.output_delta, self.weights_hidden_output.T)
        self.hidden_delta = self.hidden_error * (self.activated_hidden * (1 - self.activated_hidden))

        self.weights_input_hidden += learning_rate * np.dot(X.T, self.hidden_delta)
        self.weights_hidden_output += learning_rate * np.dot(self.activated_hidden.T, self.output_delta)

# Побудова та навчання моделі
input_size = X_train.shape[1]
hidden_size = 5
output_size = len(np.unique(y_train))

model = NeuralNetwork(input_size, hidden_size, output_size)
epochs = 1000

for epoch in range(epochs):
    output = model.forward(X_train_normalized)
    model.backward(X_train_normalized, np.eye(output_size)[y_train], output)

    if (epoch + 1) % 100 == 0:
        predictions = np.argmax(model.forward(X_train_normalized), axis=1)
        accuracy = np.mean(predictions == y_train)
        print(f"Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy:.4f}")

# Оцінка моделі на тестовому наборі
predictions_test = np.argmax(model.forward(X_test_normalized), axis=1)
accuracy_test = np.mean(predictions_test == y_test)
print(f"Accuracy on Test Set: {accuracy_test:.4f}")