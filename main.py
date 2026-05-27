# ============================================================
# ОБЩЕЕ ЗАДАНИЕ (закомментировано)
# ============================================================
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# from sklearn.datasets import make_blobs
# import pandas as pd
# dataset, classes = make_blobs(n_samples=200, n_features=2, centers=1,
#                                cluster_std=1.0, random_state=0)
# df = pd.DataFrame(dataset, columns=['var1', 'var2'])
# print("=" * 60)
# print("НАБОР ДАННЫХ (один нормально распределённый кластер):")
# print(df.head(10))
# print(f"Всего точек: {len(df)}")
#
# from yellowbrick.cluster import KElbowVisualizer
# from sklearn.cluster import KMeans
# model = KMeans(n_clusters=4, random_state=0)
# visualizer = KElbowVisualizer(model, k=(2, 11), force_model=True)
# visualizer.fit(df)
# visualizer.fig.savefig('chart2.jpeg')
# plt.clf()
# optimal_k = visualizer.elbow_value_
# print(f"МЕТОД ЛОКТЯ: оптимальное число кластеров k = {optimal_k}")
#
# from sklearn.metrics import silhouette_score
# km_check = KMeans(n_clusters=optimal_k, init='k-means++', random_state=0).fit(df)
# sil_score = silhouette_score(df, km_check.labels_)
# print(f"Silhouette Score для k={optimal_k}: {sil_score:.4f}")
# if sil_score < 0.5:
#     best_k, best_sil = optimal_k, sil_score
#     for k in range(2, 11):
#         km_tmp = KMeans(n_clusters=k, init='k-means++', random_state=0).fit(df)
#         s = silhouette_score(df, km_tmp.labels_)
#         if s > best_sil:
#             best_sil, best_k = s, k
#     optimal_k = best_k
#
# from collections import Counter
# kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=0).fit(df)
# print(kmeans.labels_)
# print(kmeans.cluster_centers_)
# print(kmeans.inertia_)
# print(kmeans.n_iter_)
# print(Counter(kmeans.labels_))
#
# labels_str = kmeans.labels_.astype(str)
# sns.scatterplot(data=df, x='var1', y='var2', hue=labels_str, palette='tab10')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
#             marker="X", c="r", s=80, label="centroids", zorder=5)
# plt.title(f"KMeans кластеризация (k={optimal_k})")
# plt.legend()
# plt.savefig('chart3.jpeg')


# ============================================================
# ИНДИВИДУАЛЬНОЕ ЗАДАНИЕ
# Микросервис кластеризации данных методом K-средних (KMeans)
# Массив M: 3 параметра, сгенерированных по варианту:
#   param1 ∈ [-10; 1], param2 ∈ [1; 2], param3 ∈ {отриц., полож.}
# ============================================================


# =============================================================================
# БЛОК ИМПОРТА БИБЛИОТЕК
# Подключение всех необходимых пакетов до начала работы
# =============================================================================
import matplotlib                  # базовая библиотека построения графиков
matplotlib.use('Agg')              # бэкенд Agg: рендеринг без дисплея, вывод в файл;
                                   # должен быть вызван ДО импорта matplotlib.pyplot
import matplotlib.pyplot as plt    # pyplot — интерфейс создания и сохранения фигур
from mpl_toolkits.mplot3d import Axes3D  # Axes3D — поддержка трёхмерных графиков

import numpy as np                 # numpy — генерация числовых массивов и математика
import pandas as pd                # pandas — табличное представление данных (DataFrame)
from collections import Counter    # Counter — подсчёт числа точек в каждом кластере

from sklearn.cluster import KMeans           # KMeans — алгоритм кластеризации K-средних
from sklearn.metrics import silhouette_score # silhouette_score — метрика качества кластеризации


# =============================================================================
# БЛОК ВВОДА ДАННЫХ
# Генерация массива M из 200 записей с тремя параметрами по варианту:
#   param1 — равномерно из диапазона [-10; 1]
#   param2 — равномерно из диапазона [1; 2]
#   param3 — может принимать отрицательные и положительные значения [-5; 5]
# =============================================================================
np.random.seed(42)   # фиксация генератора случайных чисел для воспроизводимости
N = 200              # количество записей в массиве M

# np.random.uniform(low, high, size) — генерирует равномерно распределённые
# числа в заданном диапазоне [low; high) для каждого из трёх параметров
param1 = np.random.uniform(-10, 1, N)   # первый параметр:  диапазон [-10; 1]
param2 = np.random.uniform(1,  2, N)    # второй параметр:  диапазон [1; 2]
param3 = np.random.uniform(-5, 5, N)    # третий параметр:  отрицательные и положительные значения

# np.column_stack — объединяет три одномерных массива в матрицу (N, 3)
M = np.column_stack((param1, param2, param3))

# pd.DataFrame — преобразует массив NumPy в таблицу с именованными столбцами
df = pd.DataFrame(M, columns=['param1', 'param2', 'param3'])

print("=" * 60)
print("БЛОК ВВОДА ДАННЫХ — массив M (первые 10 строк):")
print(df.head(10))
print(f"\nВсего записей: {len(df)}")
print(f"param1: [{df['param1'].min():.2f}; {df['param1'].max():.2f}]")
print(f"param2: [{df['param2'].min():.2f}; {df['param2'].max():.2f}]")
print(f"param3: [{df['param3'].min():.2f}; {df['param3'].max():.2f}]")


# =============================================================================
# БЛОК ОПРЕДЕЛЕНИЯ ОПТИМАЛЬНОГО ЧИСЛА КЛАСТЕРОВ (МЕТОД ЛОКТЯ — ELBOW)
# Для каждого k из диапазона 2–10 обучаем KMeans и записываем инерцию (WCSS).
# «Локоть» кривой — точка, после которой снижение инерции резко замедляется.
# =============================================================================
inertia_values = []   # список значений WCSS для каждого k
K_range = range(2, 11)

for k in K_range:
    # KMeans.fit — обучает модель для текущего k:
    # итеративно перемещает центроиды до минимизации WCSS
    km = KMeans(n_clusters=k, init='k-means++', random_state=0)
    km.fit(df)
    # km.inertia_ — сумма квадратов расстояний от точек до их центроидов
    inertia_values.append(km.inertia_)

# Построение и сохранение графика Elbow
plt.figure()
plt.plot(list(K_range), inertia_values, marker='o')  # кривая инерции
plt.title('Метод локтя (Elbow Method)')
plt.xlabel('Число кластеров k')
plt.ylabel('Inertia (WCSS)')
plt.savefig('chart_elbow.jpeg')   # сохранение графика Elbow в файл
plt.clf()                          # plt.clf() — очистка фигуры перед следующим графиком

print("\n" + "=" * 60)
print("БЛОК ELBOW — инерция по значениям k:")
for k, iner in zip(K_range, inertia_values):
    print(f"  k={k:>2}  WCSS={iner:.4f}")


# =============================================================================
# БЛОК ОЦЕНКИ КАЧЕСТВА КЛАСТЕРИЗАЦИИ (МЕТОД СИЛУЭТА — SILHOUETTE)
# silhouette_score показывает, насколько точка «похожа» на свой кластер
# по сравнению с соседними. Значение < 0.5 → данные плохо кластеризованы.
# В таком случае выбираем k с максимальным коэффициентом силуэта.
# =============================================================================
print("\n" + "=" * 60)
print("БЛОК SILHOUETTE — поиск оптимального k:")

best_k   = 2       # лучшее k по коэффициенту силуэта
best_sil = -1      # лучший коэффициент (инициализируем минимально возможным)

for k in K_range:
    km_tmp = KMeans(n_clusters=k, init='k-means++', random_state=0).fit(df)
    # silhouette_score — вычисляет средний коэффициент силуэта по всем точкам;
    # принимает данные и массив меток кластеров
    s = silhouette_score(df, km_tmp.labels_)
    print(f"  k={k:>2}  Silhouette={s:.4f}")
    if s > best_sil:
        best_sil = s
        best_k   = k

optimal_k = best_k
print(f"\nОптимальное k (по Silhouette): {optimal_k}  (score={best_sil:.4f})")


# =============================================================================
# БЛОК ОБУЧЕНИЯ МОДЕЛИ КЛАСТЕРИЗАЦИИ
# Финальный запуск KMeans с оптимальным числом кластеров
# =============================================================================

# KMeans параметры:
#   n_clusters=optimal_k — число кластеров, найденное методами Elbow/Silhouette
#   init='k-means++'     — умная инициализация (снижает риск локального минимума)
#   random_state=0       — фиксация для воспроизводимости
# .fit(df) — обучает модель: вычисляет центроиды и назначает метки точкам
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=0).fit(df)


# =============================================================================
# БЛОК ВЫВОДА РЕЗУЛЬТАТОВ КЛАСТЕРИЗАЦИИ
# Вывод в консоль всех требуемых характеристик модели
# =============================================================================
print("\n" + "=" * 60)
print("БЛОК РЕЗУЛЬТАТОВ КЛАСТЕРИЗАЦИИ:")

# kmeans.labels_ — массив длиной N; каждый элемент — номер кластера точки
print("\nПрогнозируемые кластеры для каждой точки (labels_):")
print(kmeans.labels_)

# kmeans.cluster_centers_ — массив координат центроидов формы (k, 3)
print("\nКоординаты центроидов кластеров:")
for i, c in enumerate(kmeans.cluster_centers_):
    print(f"  Кластер {i}: param1={c[0]:.4f}, param2={c[1]:.4f}, param3={c[2]:.4f}")

# kmeans.inertia_ — внутрикластерная сумма квадратов расстояний (WCSS)
print(f"\nВнутрикластерная сумма квадратов (Inertia): {kmeans.inertia_:.4f}")

# kmeans.n_iter_ — количество итераций алгоритма до сходимости
print(f"Количество итераций K-средних: {kmeans.n_iter_}")

# Counter — словарь {метка_кластера: количество_точек}
print("\nРазмер каждого кластера:")
for cid, size in sorted(Counter(kmeans.labels_).items()):
    print(f"  Кластер {cid}: {size} точек")


# =============================================================================
# БЛОК ВИЗУАЛИЗАЦИИ
# Трёхмерная диаграмма рассеяния с обозначением кластеров и центроидов.
# Используем 3D-график, так как массив M имеет три параметра.
# =============================================================================

# kmeans.labels_.astype(str) — преобразование меток в строки для правильного
# выбора цвета: без этого matplotlib интерпретирует метки как непрерывную шкалу
labels = kmeans.labels_

# Палитра из 10 дискретных цветов для кластеров
colors = plt.cm.tab10(np.linspace(0, 1, optimal_k))
color_map = {i: colors[i] for i in range(optimal_k)}
point_colors = [color_map[lbl] for lbl in labels]

fig = plt.figure(figsize=(10, 7))
# Axes3D — трёхмерные оси для отображения данных с тремя параметрами
ax = fig.add_subplot(111, projection='3d')

# ax.scatter — наносит точки данных в трёхмерном пространстве;
#   c=point_colors — цвет каждой точки определяется предсказанной меткой кластера
ax.scatter(df['param1'], df['param2'], df['param3'],
           c=point_colors, s=30, alpha=0.7)

# Нанесение центроидов поверх точек данных
for i, c in enumerate(kmeans.cluster_centers_):
    # marker='X', s=150 — центроиды выделяются крупным маркером X
    ax.scatter(c[0], c[1], c[2],
               color=color_map[i], marker='X', s=150,
               edgecolors='black', linewidths=0.8,
               label=f'Центроид {i}', zorder=5)

ax.set_xlabel('param1  [-10; 1]')    # метка оси X — первый параметр
ax.set_ylabel('param2  [1; 2]')      # метка оси Y — второй параметр
ax.set_zlabel('param3  [-5; 5]')     # метка оси Z — третий параметр
ax.set_title(f'KMeans 3D кластеризация (k={optimal_k})')
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig('chart_individual.jpeg', dpi=120)  # сохранение 3D-диаграммы в файл
print("\nГрафик сохранён в chart_individual.jpeg")
