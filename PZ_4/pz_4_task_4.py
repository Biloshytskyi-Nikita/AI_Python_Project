import random
import matplotlib.pyplot as plt

# Функція активації, яка визначає, який вихід має бути для даного вхідного значення
def activation_function(x):
    return 1 if x >= 0 else 0

# Клас персептрону, який представляє собою просту нейронну мережу з одним виходом
class Perceptron:
    def __init__(self, num_inputs, activation_function):
        # Ініціалізація ваг та зсуву з випадковими значеннями
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(-1, 1)
        self.activation_function = activation_function

    # Проходження вперед через персептрон
    def forward(self, inputs):
        # Обчислення зваженої суми вхідних значень та ваг, додавання зсуву та застосування функції активації
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        return self.activation_function(weighted_sum)

    # Оновлення ваг під час навчання за правилом навчання персептрону
    def update_weights(self, inputs, target, learning_rate):
        prediction = self.forward(inputs)  # Отримання передбачення персептрону для вхідних даних
        error = target - prediction  # Обчислення помилки передбачення
        # Оновлення ваг на основі помилки передбачення та швидкості навчання
        self.weights = [w + learning_rate * error * x for w, x in zip(self.weights, inputs)]
        self.bias += learning_rate * error  # Оновлення зсуву

# Цільова функція, яка визначає лінійну розділову пряму
def target_function(x):
    return 3*x - 1.5

# Генерація навчальних даних для навчання персептрону
num_points = 100
training_data = []
for _ in range(num_points):
    x = random.uniform(-3, 3)  # Генерація випадкового значення x
    y = random.uniform(-3, 3)  # Генерація випадкового значення y
    # Визначення мітки класу на основі того, чи знаходиться точка вище або нижче від лінійної функції
    label = 1 if y > target_function(x) else 0
    training_data.append(([x, y], label))  # Додавання пари вхідних даних та мітки до навчального набору даних

# Створення об'єкту персептрону з чотирма входами та функцією активації
perceptron = Perceptron(num_inputs=4, activation_function=activation_function)

learning_rate = 0.1  # Швидкість навчання
epochs = 100  # Кількість епох навчання

# Навчання персептрону на навчальних даних
for epoch in range(epochs):
    for inputs, label in training_data:
        perceptron.update_weights(inputs, label, learning_rate)  # Оновлення ваг персептрону

# Відображення лінійної розділової прямої та навчальних даних
x_values = [-3, 3]
y_values = [target_function(x) for x in x_values]
plt.plot(x_values, y_values, label='Target Function')
for inputs, label in training_data:
    color = 'red' if label == 1 else 'blue'  # Встановлення кольору в залежності від мітки класу
    plt.scatter(*inputs[:2], color=color)  # Відображення точок навчального набору даних
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Linear Separation of Two Classes')
plt.legend()
plt.show()

# Тестування персептрону на всіх можливих точках та відображення результатів класифікації
test_inputs = [[x, y] for x in range(-3, 4) for y in range(-3, 4)]
predictions = [perceptron.forward([x1, x2, 3*x1 - 1.5, 3*x2 - 1.5]) for x1, x2 in test_inputs]

for inputs, prediction in zip(test_inputs, predictions):
    color = 'red' if prediction == 1 else 'blue'  # Встановлення кольору в залежності від передбаченого класу
    plt.scatter(*inputs[:2], color=color)  # Відображення точок з використанням передбаченого класу
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Perceptron Classification')
plt.show()