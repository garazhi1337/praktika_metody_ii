import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl

#1) Построить нечеткую базу знаний (использовать не менее 3
#лингвистических переменных) для задачи выбора новой посуды (учитывая
#соотношение цены, качества и привлекательность), проверить ее на полноту и
#произвести нечеткий вывод для конкретных значений (выбрать случайным
#образом).

price = ctrl.Antecedent(np.arange(200, 5000, 1), 'price')
quality = ctrl.Antecedent(np.arange(0, 100, 1), 'quality')
design = ctrl.Antecedent(np.arange(0, 10, 1), 'design')  # новый входной параметр
attractiveness = ctrl.Consequent(np.arange(1, 10, 1), 'attractiveness')

attractiveness.automf(names=['низкая', 'средняя', 'высокая'])

# Функции принадлежности для цены
price['низкая'] = fuzz.trapmf(price.universe, [200, 800, 1400, 2000])
price['средняя'] = fuzz.trapmf(price.universe, [1600, 2200, 2800, 3400])
price['высокая'] = fuzz.trapmf(price.universe, [3000, 3600, 4200, 5000])

# Функции принадлежности для качества
quality['низкое'] = fuzz.trapmf(quality.universe, [0, 0, 20, 35])
quality['среднее'] = fuzz.trapmf(quality.universe, [30, 50, 60, 75])
quality['высокое'] = fuzz.trapmf(quality.universe, [70, 80, 90, 100])

# Функции принадлежности для дизайна (новый параметр)
design['плохой'] = fuzz.trapmf(design.universe, [0, 0, 2, 4])
design['средний'] = fuzz.trapmf(design.universe, [3, 4, 5, 7])
design['хороший'] = fuzz.trapmf(design.universe, [6, 7, 8, 10])

# Функции принадлежности для привлекательности
attractiveness['низкая'] = fuzz.trapmf(attractiveness.universe, [0, 1, 2, 4])
attractiveness['средняя'] = fuzz.trapmf(attractiveness.universe, [3, 5, 6, 8])
attractiveness['высокая'] = fuzz.trapmf(attractiveness.universe, [7, 8, 9, 10])

#27 правил для всех комбинаций
rule1 = ctrl.Rule(price['низкая'] & quality['низкое'] & design['плохой'], attractiveness['низкая'])
rule2 = ctrl.Rule(price['низкая'] & quality['низкое'] & design['средний'], attractiveness['средняя'])
rule3 = ctrl.Rule(price['низкая'] & quality['низкое'] & design['хороший'], attractiveness['средняя'])

rule4 = ctrl.Rule(price['низкая'] & quality['среднее'] & design['плохой'], attractiveness['средняя'])
rule5 = ctrl.Rule(price['низкая'] & quality['среднее'] & design['средний'], attractiveness['средняя'])
rule6 = ctrl.Rule(price['низкая'] & quality['среднее'] & design['хороший'], attractiveness['высокая'])

rule7 = ctrl.Rule(price['низкая'] & quality['высокое'] & design['плохой'], attractiveness['средняя'])
rule8 = ctrl.Rule(price['низкая'] & quality['высокое'] & design['средний'], attractiveness['высокая'])
rule9 = ctrl.Rule(price['низкая'] & quality['высокое'] & design['хороший'], attractiveness['высокая'])

# Правила для средней цены
rule10 = ctrl.Rule(price['средняя'] & quality['низкое'] & design['плохой'], attractiveness['низкая'])
rule11 = ctrl.Rule(price['средняя'] & quality['низкое'] & design['средний'], attractiveness['низкая'])
rule12 = ctrl.Rule(price['средняя'] & quality['низкое'] & design['хороший'], attractiveness['средняя'])

rule13 = ctrl.Rule(price['средняя'] & quality['среднее'] & design['плохой'], attractiveness['средняя'])
rule14 = ctrl.Rule(price['средняя'] & quality['среднее'] & design['средний'], attractiveness['средняя'])
rule15 = ctrl.Rule(price['средняя'] & quality['среднее'] & design['хороший'], attractiveness['высокая'])

rule16 = ctrl.Rule(price['средняя'] & quality['высокое'] & design['плохой'], attractiveness['средняя'])
rule17 = ctrl.Rule(price['средняя'] & quality['высокое'] & design['средний'], attractiveness['высокая'])
rule18 = ctrl.Rule(price['средняя'] & quality['высокое'] & design['хороший'], attractiveness['высокая'])

# Правила для высокой цены
rule19 = ctrl.Rule(price['высокая'] & quality['низкое'] & design['плохой'], attractiveness['низкая'])
rule20 = ctrl.Rule(price['высокая'] & quality['низкое'] & design['средний'], attractiveness['низкая'])
rule21 = ctrl.Rule(price['высокая'] & quality['низкое'] & design['хороший'], attractiveness['низкая'])

rule22 = ctrl.Rule(price['высокая'] & quality['среднее'] & design['плохой'], attractiveness['низкая'])
rule23 = ctrl.Rule(price['высокая'] & quality['среднее'] & design['средний'], attractiveness['средняя'])
rule24 = ctrl.Rule(price['высокая'] & quality['среднее'] & design['хороший'], attractiveness['средняя'])

rule25 = ctrl.Rule(price['высокая'] & quality['высокое'] & design['плохой'], attractiveness['средняя'])
rule26 = ctrl.Rule(price['высокая'] & quality['высокое'] & design['средний'], attractiveness['средняя'])
rule27 = ctrl.Rule(price['высокая'] & quality['высокое'] & design['хороший'], attractiveness['высокая'])

# есть 3*3*3 разных правил, значит каждая входная терма используется хотябы 1 раз
# и выходная терма используется 1 раз при любых параметров из диапазон

attr_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9,
                                rule10, rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18,
                                rule19, rule20, rule21, rule22, rule23, rule24, rule25, rule26, rule27])
attr_simulation = ctrl.ControlSystemSimulation(attr_ctrl)

attr_simulation.input['price'] = 4000
attr_simulation.input['quality'] = 85
attr_simulation.input['design'] = 2
attr_simulation.compute()

# Вывод результата
print(f"Входные значения: price=4000 (высокая), quality=85 (высокое), design=2 (плохой)")
print(f"Результат - привлекательность: {attr_simulation.output['attractiveness']:.2f}")

# Визуализация
price.view(sim=attr_simulation)
quality.view(sim=attr_simulation)
design.view(sim=attr_simulation)
attractiveness.view(sim=attr_simulation)

plt.ioff()
plt.show()