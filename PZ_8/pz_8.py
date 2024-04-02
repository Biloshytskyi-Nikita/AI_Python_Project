import numpy as np
import matplotlib.pyplot as plt

# Визначення функцій належності для кількості втрачених клієнтів
def membership_lost_clients(x):
    low = np.maximum(0, 1 - x / 50)
    medium = np.maximum(0, np.minimum(1, (x - 20) / 30))
    high = np.maximum(0, (x - 50) / 50)
    return low, medium, high

# Визначення функцій належності для кількості нових клієнтів
def membership_new_clients(x):
    low = np.maximum(0, 1 - x / 50)
    medium = np.maximum(0, np.minimum(1, (x - 20) / 30))
    high = np.maximum(0, (x - 50) / 50)
    return low, medium, high

# Визначення функцій належності для прибутку
def membership_profit(x):
    low = np.maximum(0, 1 - x / 50)
    medium = np.maximum(0, np.minimum(1, (x - 20) / 30))
    high = np.maximum(0, (x - 50) / 50)
    return low, medium, high

# Візуалізація функцій належності
lost_clients = np.arange(0, 101, 1)
new_clients = np.arange(0, 101, 1)
profit = np.arange(0, 101, 1)

lost_low, lost_medium, lost_high = membership_lost_clients(lost_clients)
new_low, new_medium, new_high = membership_new_clients(new_clients)
profit_low, profit_medium, profit_high = membership_profit(profit)

plt.figure(figsize=(10, 6))
plt.plot(lost_clients, lost_low, 'b', linewidth=1.5, label='Low')
plt.plot(lost_clients, lost_medium, 'g', linewidth=1.5, label='Medium')
plt.plot(lost_clients, lost_high, 'r', linewidth=1.5, label='High')
plt.plot(new_clients, new_low, 'b--', linewidth=1.5)
plt.plot(new_clients, new_medium, 'g--', linewidth=1.5)
plt.plot(new_clients, new_high, 'r--', linewidth=1.5)
plt.xlabel('Number of Clients')
plt.ylabel('Membership')
plt.title('Membership functions')
plt.legend(['Low Lost Clients', 'Medium Lost Clients', 'High Lost Clients',
            'Low New Clients', 'Medium New Clients', 'High New Clients'])
plt.grid(True)

# Визначення правил для нечіткої системи виведення
def rule_output(lost, new):
    rule1 = np.fmax(lost_low[:, np.newaxis], new_high)
    rule2 = np.fmax(lost_medium[:, np.newaxis], new_medium)
    rule3 = np.fmax(lost_high[:, np.newaxis], new_low)
    return np.fmax(np.fmax(rule1, rule2), rule3)

# Визначення діапазону вхідних даних для дослідження
input_lost_clients = np.arange(0, 101, 10)
input_new_clients = np.arange(0, 101, 10)
output_profits = []

# Обчислення прибутку для кожної комбінації вхідних даних
for lost, new in zip(input_lost_clients, input_new_clients):
    output_profit = rule_output(lost, new)
    output_profits.append(output_profit)

# Візуалізація результатів
output_profits_flat = np.array(output_profits).flatten()

plt.figure(figsize=(10, 6))
plt.plot(output_profits_flat, 'ko', markersize=8)
plt.xlabel('Cases')
plt.ylabel('Profit')
plt.title('Fuzzy Competitive Ability')
plt.grid(True)
plt.show()