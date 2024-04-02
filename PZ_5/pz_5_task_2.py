import numpy as np

# Згенеруємо випадкові дані про музичні центри
np.random.seed(0)
num_samples = 1000

# Потужність (від 1 до 10)
power = np.random.randint(1, 11, size=num_samples)

# Ціна (від 100 до 1000)
price = np.random.randint(100, 1001, size=num_samples)

# Жанр (0 - рок, 1 - поп, 2 - класика)
genre = np.random.randint(0, 3, size=num_samples)

# Перевірка перших 5 записів
print("Перші 5 записів:")
print("Потужність | Ціна | Жанр")
for i in range(5):
    print(f"{power[i]}         | ${price[i]} | {genre[i]}")

# Нормалізація числових ознак
def normalize_feature(feature):
    return (feature - np.min(feature)) / (np.max(feature) - np.min(feature))

# Нормалізація потужності та ціни
power = normalize_feature(power)
price = normalize_feature(price)

# Перевірка перших 5 записів після нормалізації
print("\nПерші 5 записів після нормалізації:")
print("Потужність | Ціна | Жанр")
for i in range(5):
    print(f"{power[i]}         | ${price[i]} | {genre[i]}")

# Використовуємо попередній клас нейронної мережі для класифікації жанру музичного центру

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

# Перевірка роботи моделі
input_size = 2 # кількість параметрів музичного центру (потужність та ціна)
hidden_size = 5 # кількість прихованих нейронів
output_size = 3 # кількість класів жанрів музичних центрів

# Навчання моделі
epochs = 1000
model = NeuralNetwork(input_size, hidden_size, output_size)
output = model.forward(np.array([power, price]).T)
for epoch in range(epochs):
    # Пряме поширення та зворотне поширення для кожного навчального прикладу
    for i in range(len(power)):
        X = np.array([power[i], price[i]]).reshape(1, -1)  # вхідний приклад
        y = np.zeros((1, output_size))  # очікуваний вихід
        y[0, genre[i]] = 1  # встановлюємо відповідний жанр в одиницю
        output = model.forward(np.array([power, price]).T)  # пряме поширення
        model.backward(np.array([power, price]).T, y, output)  # зворотне поширення

    # Оцінка точності моделі після кожної епохи
    if (epoch + 1) % 100 == 0:
        predictions = model.forward(np.array([power, price]).T)
        loss = np.mean(np.square(genre - np.argmax(predictions, axis=1)))
        print(f"Епоха {epoch + 1}/{epochs}, Втрата: {loss:.4f}")

# Тестування моделі
test_input = np.array([[0.6, 0.8],  # потужність: 60%, ціна: 80%
                       [0.3, 0.5],  # потужність: 30%, ціна: 50%
                       [0.9, 0.7]]) # потужність: 90%, ціна: 70%
predictions = model.forward(test_input)
print("\nПрогнози для тестових даних:")
for i in range(len(test_input)):
    genre_label = np.argmax(predictions[i])
    genre_str = ""
    if genre_label == 0:
        genre_str = "рок"
    elif genre_label == 1:
        genre_str = "поп"
    else:
        genre_str = "класика"
    print(f"Жанр музичного центру: {genre_str}, Потужність: {test_input[i, 0] * 10}, Ціна: ${test_input[i, 1] * 900 + 100}")
