import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from scipy import interpolate
from scipy.interpolate import interp1d, griddata, CubicSpline
import os
import openpyxl

# создание массивов
days_name = ['Посев',
             'Первый день',
             'Второй день',
             'Третий день',
             'Четвертый день',
             'Пятый день',
             'Шестой день',
             'Седьмой день'
            ]
data = {
    'Масса муки': np.array([5, 10, 15, 20, 20, 15, 10, 5, 10, 15, 20, 5, 10, 15, 20, 5]),
    'Масса глюкозы': np.array([20, 20, 20, 20, 15, 15, 15, 5, 5, 5, 5, 10, 10, 10, 10, 15]),
    'Биомасса': {
             0: np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
             1: np.array([0.28, 0.5, 0.2, 0.24, 0.3, 0.3, 0.23, 0.34, 0.43, 0.5, 0.32, 0.45, 0.44, 0.3, 0.26, 0.42]),
             2: np.array([0.64, 1.34, 0.77, 0.55, 0.42, 0.34, 0.46, 0.84, 0.64, 0.67, 0.4, 0.72, 0.52, 0.44, 0.38, 0.48]),
             3: np.array([1, 1.72, 1.62, 0.94, 0.64, 0.43, 0.84, 0.94, 0.97, 1.32, 0.56, 1.16, 0.67, 0.67, 0.64, 0.54]),
             4: np.array([1.06, 2.44, 2.16, 1.24, 0.94, 0.64, 1.1, 1.2, 1.15, 1.64, 0.74, 1.67, 0.73, 0.73, 0.79, 0.61]),
             5: np.array([1.2, 2.64, 2.31, 1.84, 1.06, 0.94, 1.24, 1.59, 1.33, 1.79, 0.97, 1.74, 0.84, 0.91, 0.76, 0.51]),
             6: np.array([1.25, 2.63, 2.43, 2.1, 1.19, 1.04, 1.25, 1.68, 1.39, 1.91, 1.26, 1.86, 0.97, 0.86, 0.64, 0.5]),
             7: np.array([1.34, 2.5, 2.38, 2.16, 1.21, 1.16, 1.23, 1.67, 1.5, 2.1, 1.22, 1.94, 0.98, 0.72, 0.52, 0.47])
    }
}
df = {
    'Масса муки': [],
    'Масса глюкозы': [],
    'Биомасса': []
}


#функция интерполяции
def interpolate_2d(x, y, z, method='linear'):
    # Создаем регулярную сетку для интерполяции
    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    xi, yi = np.meshgrid(xi, yi)

    # Выполняем интерполяцию
    zi = griddata((x, y), z, (xi, yi), method=method)

    return xi, yi, zi


#получение датафрейма для МО
for i in range(len(data['Биомасса'])):
    xi, yi, zi = interpolate_2d(data['Масса муки'], data['Масса глюкозы'], data['Биомасса'][i], 'cubic')
    for j in range(len(zi)):
        for k in range(len(zi[j])):
            df['Масса муки'].append(xi[j][k])
            df['Масса глюкозы'].append(yi[j][k])
            df['Биомасса'].append(zi[j][k])
    # plt.contourf(xi, yi, zi)
    # plt.scatter(data['Масса муки'], data['Масса глюкозы'], c=data['Биомасса'][i])
    # plt.colorbar()
    # plt.title(f'2D Interpolation {i + 1}')
    # plt.show()


#разделение датафрейма по дням
df = pd.DataFrame(df)
chunk_size = 10000
new_dfs = []
for i in range(0, len(df), chunk_size):
    chunk = df.iloc[i:i+chunk_size]
    new_dfs.append(chunk)


#обучение основной модели
for i in range(len(new_dfs)):
    new_dfs[i] = pd.DataFrame(new_dfs[i])
    x_train, y_train = new_dfs[i].drop('Биомасса', axis=1), new_dfs[i]['Биомасса']
    model = GradientBoostingRegressor(max_depth=7)
    model.fit(x_train, y_train)
    pred = model.predict(x_train)
    pred = pred.reshape(100, 100)
    x = new_dfs[i]['Масса муки']
    y = new_dfs[i]['Масса глюкозы']
    # x = x.values
    # y = y.values
    # x = x.reshape(100, 100)
    # y = y.reshape(100, 100)
    # plt.figure()
    # plt.contourf(x, y, pred, levels=np.linspace(0, 2.8, 100))
    # plt.colorbar()
    # plt.xlabel('Масса глюкозы, г')
    # plt.ylabel('Масса муки, г')
    # pic_name = 'Предсказанная биомасса (' + days_name[i] + ')'
    # plt.title(pic_name)
    # #plt.savefig(f'Предсказанная биомасса {i}.png')
    # plt.show()


#основная функция
def calc_biomass(M_m, M_g):
    predicts = []  # для записи наших предиктов по дням
    new_predicts = []  # для извлеченного массива предиктов
    days = [1, 2, 3, 4, 5, 6, 7]  # для интерполяции
    df_predicts = pd.DataFrame({'Масса муки': [M_m] * 7, 'Масса глюкозы': [M_g] * 7})  # датафрейм для будущей модельки
    df_predicts['День'] = days
    ans = pd.DataFrame({'Масса муки': [M_m], 'Масса глюкозы': [M_g]})
    # df_last = pd.read_excel('DataFrame(Original).xlsx')

    for i in range(len(new_dfs)):
        new_dfs[i] = pd.DataFrame(new_dfs[i])
        x_train, y_train = new_dfs[i].drop('Биомасса', axis=1), new_dfs[i]['Биомасса']
        model = GradientBoostingRegressor(max_depth=7)
        model.fit(x_train, y_train)
        pred = (model.predict(ans))
        predicts.append(pred)

    for arr in predicts:  # извлечение массива из массива
        new_predicts.extend(arr)

    new_predicts.pop(0)  # убираем нулевой день

    df_predicts['Биомасса'] = new_predicts
    ##########################
    days1 = df_predicts['День'].values
    mass = df_predicts['Биомасса'].values
    cs = CubicSpline(days1, mass)
    # Генерируем точки для гладкого графика сплайна
    x_fine = np.linspace(days1.min(), days1.max(), 100)
    y_fine = cs(x_fine)
    df_last = pd.DataFrame({'Масса муки': M_m, 'Масса глюкозы': M_g, 'День': x_fine, 'Биомасса': y_fine})
    ###########################
    # обучаем последнюю модель для графиков
    x_train, y_train = df_last.drop('Биомасса', axis=1), df_last['Биомасса']
    model = GradientBoostingRegressor(max_depth=7)
    model.fit(x_train, y_train)
    test = df_predicts.drop('Биомасса', axis=1)
    pred = model.predict(test)

    df_predicts1 = df_predicts.assign(prediction_no_i=pd.Series(pred).values)

    days_m = df_predicts1['День'].values
    mass_m = df_predicts1['prediction_no_i'].values

    #days_e = df_predicts1['День'].values
    #mass_e = df_predicts1['Биомасса'].values

    # result = pd.read_excel('DataFrame.xlsx')
    # filter_df = result[(result['Масса муки'] == M_m) & (result['Масса глюкозы'] == M_g)]
    # days_true = filter_df['День'].values[1:]
    # mass_true = filter_df['Биомасса'].values[1:]

    plt.figure(figsize=(10, 6))
    plt.title('Питательная среда')
    plt.xlabel('Дни, ед.')
    plt.ylabel('Биомасса')
    # интерполяция по 7 точкам эксперимента
    # cs_true = CubicSpline(days_true, mass_true)
    # Генерируем точки для гладкого графика сплайна
    # x_fine_true = np.linspace(days_true.min(), days_true.max(), 100)
    # y_fine_true = cs_true(x_fine_true)
    #plt.plot(x_fine_true, y_fine_true, color='black', label='Экспериментальные данные')
    #plt.scatter(days_true, mass_true, color='violet', label='Экспериментальные данные')

    # интерполяция по 7 точкам эксперимента
    # cs_e = CubicSpline(days, mass_e)
    # Генерируем точки для гладкого графика сплайна
    #     x_fine = np.linspace(days1.min(), days1.max(), 100)
    #     y_fine = cs(x_fine)
    #     x_fine_ml = np.linspace(days1.min(), days1.max(), 100)
    #     y_fine_ml = cs(x_fine)
    # plt.scatter(df_predicts['День'], df_predicts['Биомасса'], color='red', label='Машинное обучение')
    # plt.plot(days1, mass, color='green', label='Предсказанные значения без интерполяции')

    # Интерполяция по 7 точкам модели
    cs_m = CubicSpline(days_m, mass_m)
    # Генерируем точки для гладкого графика сплайна
    x_fine_m = np.linspace(days_m.min(), days_m.max(), 100)
    y_fine_m = cs_m(x_fine_m)
    plt.plot(df_last['День'], df_last['Биомасса'], color='brown', label='Предсказанные значения с интерполяцией')
    plt.scatter(days_m, mass_m, color='blue', label='Интерполированное машинное обучение')
    plt.legend()
    plt.savefig(f'График для {M_m},{M_g}.png')
    plt.show()
    print(pred)
    # Проверяем наличие файла и создаем его, если не существует
    filename = 'all_experiments.xlsx'
    if not os.path.isfile(filename):
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            df_predicts['Время'] = pd.Timestamp.now().strftime("%d-%m-%Y %H:%M:%S")
            df_predicts.to_excel(writer, sheet_name=f'{M_m} {M_g}', index=False)
    else:
        with pd.ExcelWriter(filename, engine='openpyxl', mode='a') as writer:
            df_predicts['Время'] = pd.Timestamp.now().strftime("%d-%m-%Y %H:%M:%S")
            df_predicts.to_excel(writer, sheet_name=f'{M_m} {M_g}', index=False)

if __name__ == '__main__':
    calc_biomass(15,15)