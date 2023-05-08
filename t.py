income = sum([
    1550,
])

expenses = sum([
    4200,
    151,
    239.15,
    266.54,
    65,
    268.53,
    1081,
    539.25,
    231.87,
    1000,
    285.37,
    1878,
    150.26,
    174.13 
])

subscriptions = [
    (149.99, '3Tre'),
    (79, 'Folksam'),
    (151, 'Trygghansa'),
    (65, 'Spotify'),
    (150.26, 'WoW Sub')
]

savings = [
    (500, 'Avanza'),
    (500, 'Opti'),
    (400, 'Privat'),
    (1050, 'Sommar Hyra Budget')
]

current_money = 1627.40 + 6272.79

money_at_start = current_money - income + expenses

print(f'     Start: {money_at_start:8.2f}')
print(f'    Income: {income:8.2f}')
print(f'  Expenses: {expenses:8.2f}')
print(f'       Now: {money_at_start + income - expenses:8.2f}')
print(f'Now Actual: {current_money:8.2f}')
print(f'     Error: {current_money - (money_at_start + income - expenses):8.2f}')

print()
print(f'Subscriptions:')
for cost, name in subscriptions:
    print(f'{name.rjust(12)}: {cost:7.2f}')
print(f'       Total: {sum([c for c, n in subscriptions]):7.2f}')

print()
print(f'Savings:')
for cost, name in savings:
    print(f'{name.rjust(20)}: {cost:7.2f}')
print(f'               Total: {sum([c for c, n in savings]):7.2f}')
