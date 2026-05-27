# import matplotlib.pyplot as plt

# fig, ax = plt.subplots()
# ax.plot([1, 2, 3, 4], [1, 4, 2, 5])
# plt.ylabel('some numbers')
# plt.savefig('chart1.jpeg')
# plt.clf()


# =============================================================================
# БЛОК ИМПОРТА БИБЛИОТЕК
# Подключение всех необходимых Python-пакетов для работы микросервиса
# =============================================================================
from sklearn.datasets import make_blobs      # генерация синтетических кластерных данных
from sklearn.cluster import KMeans           # алгоритм кластеризации K-средних
from sklearn.metrics import silhouette_score # метрика качества кластеризации (силуэт)
from collections import Counter              # подсчёт количества точек в каждом кластере
import pandas as pd                          # работа с табличными данными (DataFrame)
import matplotlib                            # базовая библиотека построения графиков
import matplotlib.pyplot as plt              # интерфейс построения графиков
import seaborn as sns                        # продвинутая визуализация на базе matplotlib

matplotlib.use('WebAgg')  # бэкенд WebAgg запускает встроенный HTTP-сервер matplotlib
                          # и открывает интерактивный графический интерфейс в браузере
matplotlib.rcParams['webagg.port'] = 5000          # порт HTTP-сервера для WebAgg
matplotlib.rcParams['webagg.open_in_browser'] = False  # не открывать браузер автоматически


# =============================================================================
# БЛОК ВВОДА / ГЕНЕРАЦИИ ДАННЫХ
# Создание синтетического набора данных с помощью make_blobs
# =============================================================================

# make_blobs — генерирует точки данных, сгруппированные вокруг заданных центров;
#   n_samples  — общее количество генерируемых точек;
#   n_features — число признаков (координат) каждой точки;
#   centers    — число истинных центров (кластеров) при генерации;
#   cluster_std — стандартное отклонение точек вокруг центра (разброс);
#   random_state — фиксирует генератор случайных чисел для воспроизводимости
dataset, classes = make_blobs(
    n_samples=200,
    n_features=2,
    centers=1,        # один нормально распределённый кластер (по условию задания)
    cluster_std=1.0,
    random_state=0
)

# pd.DataFrame — преобразует массив NumPy в таблицу с именованными столбцами
df = pd.DataFrame(dataset, columns=['var1', 'var2'])

print("=" * 60)
print("БЛОК ВВОДА ДАННЫХ — первые 10 строк датасета:")
print(df.head(10))          # вывод первых строк датасета в консоль
print(f"Всего точек: {len(df)}")


# =============================================================================
# БЛОК ОПРЕДЕЛЕНИЯ ОПТИМАЛЬНОГО ЧИСЛА КЛАСТЕРОВ (МЕТОД ЛОКТЯ + SILHOUETTE)
# Перебор значений k, подсчёт инерции и коэффициента силуэта для каждого k.
# Метод локтя (Elbow): ищем «излом» на кривой инерции.
# Метод силуэта (Silhouette): выбираем k с максимальным средним коэффициентом.
# =============================================================================

inertia_values    = []   # WCSS (внутрикластерная сумма квадратов) для каждого k
silhouette_values = []   # средний коэффициент силуэта для каждого k
K_range = range(2, 11)   # диапазон проверяемых значений числа кластеров

for k in K_range:
    # KMeans — создаёт модель кластеризации;
    #   n_clusters  — число искомых кластеров;
    #   init='k-means++' — умная инициализация центроидов (снижает риск локального минимума);
    #   random_state — фиксирует начальное состояние для воспроизводимости
    km = KMeans(n_clusters=k, init='k-means++', random_state=0)

    # fit — обучает модель: итеративно перемещает центроиды до сходимости
    km.fit(df)

    # km.inertia_ — сумма квадратов расстояний от каждой точки до её центроида (WCSS)
    inertia_values.append(km.inertia_)

    # silhouette_score — среднее значение коэффициента силуэта по всем точкам;
    # значение близкое к 1 означает хорошо разделённые кластеры
    silhouette_values.append(silhouette_score(df, km.labels_))

print("\n" + "=" * 60)
print("БЛОК ОПРЕДЕЛЕНИЯ ОПТИМАЛЬНОГО k (Elbow + Silhouette):")
print(f"{'k':>4} | {'Inertia (WCSS)':>16} | {'Silhouette Score':>16}")
print("-" * 44)
for k, iner, sil in zip(K_range, inertia_values, silhouette_values):
    print(f"{k:>4} | {iner:>16.4f} | {sil:>16.4f}")

# Автоматический выбор оптимального k по максимальному коэффициенту силуэта
optimal_k = list(K_range)[silhouette_values.index(max(silhouette_values))]
print(f"\nОптимальное число кластеров (по Silhouette): k = {optimal_k}")


# =============================================================================
# БЛОК ОБУЧЕНИЯ МОДЕЛИ КЛАСТЕРИЗАЦИИ
# Финальное обучение KMeans с найденным оптимальным числом кластеров
# =============================================================================

# KMeans — итоговая модель с оптимальным k, найденным методом силуэта
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=0)

# fit — обучает модель на полном датасете и сохраняет результаты в атрибутах объекта
kmeans.fit(df)


# =============================================================================
# БЛОК ВЫВОДА РЕЗУЛЬТАТОВ КЛАСТЕРИЗАЦИИ
# Печать ключевых характеристик обученной модели в консоль
# =============================================================================

print("\n" + "=" * 60)
print("БЛОК РЕЗУЛЬТАТОВ КЛАСТЕРИЗАЦИИ:")

# kmeans.labels_ — массив длиной n_samples; каждый элемент — номер кластера точки
print(f"\nПрогнозируемые кластеры (первые 20 значений):")
print(kmeans.labels_[:20])

# kmeans.cluster_centers_ — массив координат центроидов формы (k, n_features)
print(f"\nКоординаты центроидов кластеров:")
for i, center in enumerate(kmeans.cluster_centers_):
    print(f"  Кластер {i}: var1 = {center[0]:.4f},  var2 = {center[1]:.4f}")

# kmeans.inertia_ — внутрикластерная сумма квадратов (WCSS) финальной модели
print(f"\nВнутрикластерная сумма квадратов (Inertia / WCSS): {kmeans.inertia_:.4f}")

# kmeans.n_iter_ — число итераций алгоритма EM до достижения сходимости
print(f"Количество итераций K-средних: {kmeans.n_iter_}")

# Counter — словарь {метка_кластера: количество_точек}
cluster_sizes = Counter(kmeans.labels_)
print(f"\nРазмер каждого кластера:")
for cluster_id, size in sorted(cluster_sizes.items()):
    print(f"  Кластер {cluster_id}: {size} точек")


# =============================================================================
# БЛОК ВИЗУАЛИЗАЦИИ
# Построение диаграммы рассеяния с обозначением кластеров и центроидов
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"KMeans кластеризация (k = {optimal_k})", fontsize=14)

# --- График 1: кривая Elbow (зависимость инерции от числа кластеров) ---
axes[0].plot(list(K_range), inertia_values, marker='o', linewidth=2)
axes[0].set_title("Метод локтя (Elbow Method)")
axes[0].set_xlabel("Число кластеров k")
axes[0].set_ylabel("Inertia (WCSS)")
# Вертикальная линия отмечает выбранный оптимальный k
axes[0].axvline(x=optimal_k, color='red', linestyle='--', label=f'k={optimal_k}')
axes[0].legend()

# --- График 2: диаграмма рассеяния с предсказанными метками кластеров ---
# sns.scatterplot — рисует точки данных; hue окрашивает точки по метке кластера
sns.scatterplot(
    data=df,
    x='var1',
    y='var2',
    hue=kmeans.labels_.astype(str),   # hue — цвет каждой точки по номеру кластера
    palette='tab10',
    s=60,
    alpha=0.8,
    legend='full',
    ax=axes[1]
)

# plt.scatter — поверх рассеяния наносит центроиды маркером X красного цвета
axes[1].scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker='X',      # маркер X визуально выделяет центроиды среди точек данных
    c='red',
    s=150,
    zorder=5,
    label='Центроиды'
)
axes[1].set_title("Диаграмма рассеяния кластеров")
axes[1].set_xlabel("var1")
axes[1].set_ylabel("var2")
axes[1].legend(title='Кластер', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()

# Сохранение итогового графика в файл JPEG
plt.savefig('chart3.jpeg', dpi=120)
print("\nГрафик сохранён в chart3.jpeg")

# plt.show() — отображает оба графика в графическом окне GUI
plt.show()
