from scipy.stats import binom_test
wins = int(input('Количество выигрышей бота = '))
ngames = int(input('Количество игр = '))
p = 0.5
bt = binom_test(wins, ngames, p)
print('Бином тест = ', bt*100, '%')
