import math

def calculate_sphere_volume(radius):
    volume = (4/3) * math.pi * radius**3
    return volume

def main():
    # Запит користувача на введення радіуса шару
    radius = float(input("Будь ласка, введіть радіус шару: "))
    
    # Обчислення об'єму шару за допомогою функції
    volume = calculate_sphere_volume(radius)
    
    # Виведення результату
    print("Об'єм шару з радіусом", radius, "дорівнює", volume)

if __name__ == "__main__":
    main()