#!/usr/bin/env python
# coding: utf-8

# ### Оптимизация бизнес-модели сервиса аренды самокатов  
# 
# ## Описание проекта  
# 
# **Клиент:** Сервис аренды самокатов GoFast  
# 
# **Цель:**  
# Анализ поведения пользователей для оптимизации тарифных планов и увеличения прибыли.  
# 
# **Ключевые вопросы:**  
# 1. Как подписка Ultra влияет на частоту и длительность поездок?  
# 2. В каких городах пользователи чаще выбирают платную подписку?  
# 3. Как возраст пользователей коррелирует с типом подписки?  
# 4. Какие гипотезы могут увеличить конверсию в платную подписку?  
# 
# ## Данные  
# 
# **Основные таблицы:**  
# 
# ### `users_go.csv`  
# - `user_id`, `name`, `age`, `city`  
# - `subscription_type` (free/ultra)  
# 
# ### `rides_go.csv`  
# - `user_id`, `distance` (метры)  
# - `duration` (минуты), `date`  
# 
# ### `subscriptions_go.csv`  
# - Тарифные параметры:  
#   - `minute_price` (6₽ для Ultra, 8₽ для free)  
#   - `start_ride_price` (0₽ для Ultra, 50₽ для free)  
#   - `subscription_fee` (199₽/мес для Ultra)  
# 
# ## Методология  
# 
# 1. **Объединение данных** по `user_id`  
# 2. **Расчет метрик:**  
#    - ARPU (средний доход с пользователя)  
#    - LTV (пожизненная ценность)  
#    - Конверсия в подписку Ultra  
# 3. **A/B-тестирование:**  
#    - Сравнение поведения пользователей на разных тарифах  
# 4. **Геоанализ:**  
#    - Распределение спроса по городам  

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# #### Шаг 1. Загрузка данных

# In[2]:


try:
    data_users = pd.read_csv('/datasets/users_go.csv')
except:
    data_users = pd.read_csv('https://code.s3.yandex.net/datasets/users_go.csv')


# In[3]:


data_users


# In[4]:


data_users.info()


# In[5]:


data_users.describe()


# In[6]:


try:
    data_rides = pd.read_csv('/datasets/rides_go.csv')
except:
    data_rides = pd.read_csv('https://code.s3.yandex.net/datasets/rides_go.csv')


# In[7]:


data_rides


# In[8]:


data_rides.info()


# In[9]:


data_rides.describe()


# In[10]:


try:
    data_subscriptions = pd.read_csv('/datasets/subscriptions_go.csv')
except:
    data_subscriptions = pd.read_csv('https://code.s3.yandex.net/datasets/subscriptions_go.csv')


# In[11]:


data_subscriptions


# In[12]:


data_subscriptions.info()


# У нас есть три датафрейма, и для дальнейшей работы с ними необходимо обработать их данные. С помощью метода info() мы видим, что пропусков в данных нет. Далее проведем обработку дубликатов.

# #### Шаг 2. Предобработка данных

# In[13]:


data_rides['date'] = pd.to_datetime(data_rides['date'])


# In[14]:


data_rides.info()


# In[15]:


# data_rides['month'] = data_rides['date'].dt.month
data_rides['month'] = pd.to_datetime(data_rides['date']).astype('datetime64[M]')
data_rides['month'] = pd.to_datetime(data_rides['date']).dt.month


# Использовав метод info() я не обнаружил пропуски. Изучив данные в трёх датафреймах, я считаю то что нужно проверить на дубликаты следующие данные:
# 1. Датафрейм data_users, столбец user_id
# 2. Датафрейм data_rides, столбец distance
# 3. Датафрейм data_rides, столбец duration

# In[16]:


data_users.duplicated().sum()


# Смотрим дубликаты

# In[17]:


data_users.loc[data_users['user_id'].duplicated(keep=False)].sort_values(by=['user_id'])


# Убедились, то что это и в самом деле дубликаты. Можем их удалить.

# In[18]:


data_users = data_users.drop_duplicates()
data_users.duplicated().sum()


# In[19]:


data_rides.duplicated().sum()


# Полностью дубликатов сток нет. Посмотрим по конкретным столбцам.

# In[20]:


data_rides['distance'].duplicated().sum()


# In[21]:


data_rides['duration'].duplicated().sum()


# Обнаружено 94 дубликата в столбце с продолжительностью поездки. Значения в этом столбце имеют 6 знаков после запятой. Такое количество совпадающих значений вызывает подозрение, давайте рассмотрим их ближе.

# In[22]:


data_rides.loc[data_rides['duration'].duplicated(keep=False)].sort_values(by=['user_id'])


# Мы видим, что это на самом деле разные поездки, которые содержат корректные данные всех столбцов, кроме "duration". Мы можем предположить, что продолжительность так мала из-за технических проблем во время поездок. На мой взгляд, удалять эти данные не стоит, потому что мы сможем воспользоваться информацией из других столбцов в будущем.

# В ходе предобработки данных я удалил дубликаты из датафрейма data_users с данными об одних и тех же пользователях. Дубликаты из датафрейма data_rides не были удалены по причине того, что появившиеся там дубликаты, скорее всего, вызваны техническим сбоем. Строки с дубликатами в этом датафрейме содержат полезные данные для дальнейшего анализа.

# #### Шаг 3. Исследовательский анализ данных

# In[23]:


data_users.pivot_table(index='city', values='user_id', aggfunc='count')


# In[24]:


plt.figure(figsize=(10, 6)) 

data_users.pivot_table(index='city', 
                       values='user_id', 
                       aggfunc='count').sort_values(by='user_id', 
                                                    ascending=False).plot(y='user_id', 
                                                                          kind='bar', 
                                                                          grid=True, 
                                                                          legend=False)

plt.xlabel('Город')
plt.ylabel('Количество пользователей')
plt.title('Количество пользователей по городам')

plt.xticks(rotation=45, ha='right')

plt.show()


# Больше всего пользователей самокатов в Екатеринбурге. Меньше всего — в Тюмени. Но разница в количестве пользователей незначительная, их количество примерно одинаковое.

# In[25]:


data_users.pivot_table(index='subscription_type', values='user_id', aggfunc='count')


# In[26]:


plt.figure(figsize=(10, 6)) 

data_users.pivot_table(index='subscription_type', 
                       values='user_id', 
                       aggfunc='count').plot(y='user_id', 
                                           kind='bar', 
                                           grid=True, 
                                           legend=False)

plt.xlabel('Наличие подписки')
plt.ylabel('Количество пользователей')
plt.title('Количество пользователей по подпискам')

plt.xticks(rotation=45, ha='right')

plt.show()


# Пользователей с подпиской 'ultra' меньше чем пользователей без подписки, но разница не критичная.

# In[27]:


total_free_users = data_users[data_users['subscription_type'] == 'free']['user_id'].count()
total_ultra_users = data_users[data_users['subscription_type'] == 'ultra']['user_id'].count()

total_users = total_free_users + total_ultra_users

share_free_users = total_free_users / total_users
share_ultra_users = total_ultra_users / total_users

print(f"Доля пользователей без подписки: {share_free_users:.2%}")
print(f"Доля пользователей с подпиской: {share_ultra_users:.2%}")


# In[28]:


labels = ['Без подписки', 'С подпиской']
sizes = [share_free_users, share_ultra_users]
colors = ['lightcoral', 'lightskyblue']
explode = (0.1, 0)

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')

plt.show()


# In[29]:


data_users.pivot_table(index='age', values='user_id', aggfunc='count').plot(y='user_id', kind='bar')


# Возраст пользователей имеет нормальное распределение
# 

# In[30]:


data_rides.pivot_table(index='user_id', values='distance').describe()


# In[31]:


# data_rides.pivot_table(index='user_id', values='distance').plot(kind='bar', alpha=0.7)
# plt.xticks([])

data_rides.pivot_table(index='user_id', values='distance').hist(bins=100)


# Среднее и медианное расстояние, которое пользователь преодолел за одну поездку, оказалось близким к 3 км. Распределение дистанций, преодолеваемых пользователями, стремится к нормальному, что видно по гистограмме.

# In[32]:


data_rides.pivot_table(index='user_id', values='duration').describe()


# In[33]:


# data_rides.pivot_table(index='user_id', values='duration').plot(kind='bar', alpha=0.7)
# plt.xticks([])

data_rides.pivot_table(index='user_id', values='duration').hist(bins=100)


# Среднее и медианное время продолжительности поездки оказалось близким к 18 минутам. Распределение времени использования самоката стремится к нормальному, что видно по гистограмме.

# Распределение пользователей самокатов по городам показывает, что больше всего пользователей находится в Екатеринбурге, а меньше всего в Тюмени. Однако, разница в количестве пользователей между этими городами незначительна, их количество примерно одинаковое.
# 
# Количество пользователей с подпиской 'ultra' немного меньше, чем количество пользователей без подписки, но разница не критична. Доля пользователей без подписки составляет 54.43%, в то время как доля пользователей с подпиской 'ultra' - 45.57%.
# 
# Возраст пользователей имеет нормальное распределение, что может быть полезной информацией при анализе и разработке стратегии маркетинга.
# 
# Среднее и медианное расстояние, которое пользователь преодолевает за одну поездку, составляет примерно 3 километра. Гистограмма распределения дистанций указывает на то, что они стремятся к нормальному распределению, что может быть важным для оптимизации сервиса.
# 
# Среднее и медианное время продолжительности поездки составляет около 18 минут. Распределение времени использования самоката также стремится к нормальному, что видно по гистограмме. Эти данные могут быть полезными для планирования ресурсов и обслуживания клиентов.

# #### Шаг 4. Объединение данных

# In[34]:


df = data_users.merge(data_rides, on='user_id', how='outer')
df = df.merge(data_subscriptions, on='subscription_type', how='outer')
df


# In[35]:


df.describe()


# In[36]:


df.info()


# In[37]:


df_unsub = df[df['subscription_fee'] == 0]
df_sub = df[df['subscription_fee'] == 199]


# In[38]:


# df_unsub.pivot_table(index='user_id', values='distance').plot(kind='bar', alpha=0.7)
# plt.xticks([])

df_unsub.pivot_table(index='user_id', values='distance').hist(bins=100)


# In[39]:


df_unsub.pivot_table(index='user_id', values='distance').describe()


# Средняя дистанция за одну поезду у пользователей без подписки - 3км

# In[40]:


# df_unsub.pivot_table(index='user_id', values='duration').plot(kind='bar', alpha=0.7)
# plt.xticks([])
df_unsub.pivot_table(index='user_id', values='duration').hist(bins=100)


# In[41]:


df_unsub.pivot_table(index='user_id', values='duration').describe()


# Среднее время одной поезди у пользователей без подписки - 17.5 минут

# In[42]:


# df_sub.pivot_table(index='user_id', values='distance').plot(kind='bar', alpha=0.7)
# plt.xticks([])
df_sub.pivot_table(index='user_id', values='distance').hist(bins=100)


# In[43]:


df_sub.pivot_table(index='user_id', values='distance').describe()


# Средняя дистанция за одну поезду у пользователей без подписки - 3.1 км

# In[44]:


# df_sub.pivot_table(index='user_id', values='duration').plot(kind='bar', alpha=0.7)
# plt.xticks([])

df_sub.pivot_table(index='user_id', values='duration').hist(bins=100)


# In[45]:


df_sub.pivot_table(index='user_id', values='duration').describe()


# Среднее время одной поезди у пользователей без подписки - 18.5 минут

# In[46]:


alpha = 0.5

plt.hist(df_unsub['distance'], bins=100, alpha=alpha, label='Без подписки', color='blue')
plt.hist(df_sub['distance'], bins=100, alpha=alpha, label='С подпиской', color='green')

plt.xlabel('Расстояние (км)')
plt.ylabel('Частота')
plt.title('Распределение дистанций для пользователей с и без подписки')
plt.legend(loc='upper right')

plt.show()


# In[47]:


alpha = 0.5

plt.hist(df_unsub['duration'], bins=100, alpha=alpha, label='Без подписки', color='blue')
plt.hist(df_sub['duration'], bins=100, alpha=alpha, label='С подпиской', color='green')

plt.xlabel('Время (мин)')
plt.ylabel('Частота')
plt.title('Распределение времени для пользователей с и без подписки')
plt.legend(loc='upper right')

plt.show()


# Средняя дистанция за одну поездку у пользователей с подпиской и без подписки примерно одинакова и составляет около 3 километров. Это говорит о том, что пользователи, независимо от типа подписки, часто выбирают поездки примерно одинаковой длины.
# 
# Среднее время одной поездки также близко для пользователей с подпиской и без нее, составляя около 18 минут. Это может указывать на то, что в среднем пользователи тратят примерно одинаковое количество времени на поездки, независимо от подписки.
# 
# Следовательно, с точки зрения дистанции и времени одной поездки, подписка не сильно влияет на поведение пользователей, и средние значения близки между двумя группами.
# 
# Дополнительные статистические тесты могут быть проведены, чтобы более подробно изучить различия между этими группами и убедиться в статистической значимости результатов.

# #### Шаг 5. Подсчёт выручки

# In[48]:


df['duration'] = np.ceil(df['duration'])


# In[49]:


aggregated_df = df.groupby(['user_id', 'month']).agg(
    total_distance=pd.NamedAgg(column='distance', aggfunc='sum'),
    total_trips=pd.NamedAgg(column='distance', aggfunc='count'),
    total_duration=pd.NamedAgg(column='duration', aggfunc='sum')
).reset_index()
aggregated_df


# In[50]:


aggregated_df = df.groupby(['user_id', 'month']).agg(
    total_distance=pd.NamedAgg(column='distance', aggfunc='sum'),
    total_trips=pd.NamedAgg(column='distance', aggfunc='count'),
    total_duration=pd.NamedAgg(column='duration', aggfunc='sum'),
    minute_price = pd.NamedAgg(column='minute_price', aggfunc='mean'),
    start_ride_price = pd.NamedAgg(column='start_ride_price', aggfunc='mean'),
    subscription_fee = pd.NamedAgg(column='subscription_fee', aggfunc='mean')
).reset_index()

# aggregated_df['total_duration'] = np.ceil(aggregated_df['total_duration'])

aggregated_df['revenue'] = aggregated_df['minute_price'] * aggregated_df['total_duration'] + aggregated_df['start_ride_price'] * aggregated_df['total_trips'] + aggregated_df['subscription_fee']

# aggregated_df = aggregated_df.drop(columns=['minute_price', 'start_ride_price', 'subscription_fee'])
aggregated_df


# #### Шаг 6. Проверка гипотез

# Нулевая гипотеза (H0): Средняя продолжительность поездок пользователей с подпиской и без подписки одинакова.
# 
# Альтернативная гипотеза (H1):  Средняя продолжительность поездок пользователей с подпиской больше, чем у пользователей без подписки.

# In[51]:


results = stats.ttest_ind(df_unsub['duration'], df_sub['duration'], alternative='greater')

alpha = 0.05

if results.pvalue < alpha:
    print("Есть статистически значимая разница в продолжительности поездок между пользователями с подпиской и без подписки.")
else:
    print("Нет статистически значимой разницы в продолжительности поездок между пользователями с подпиской и без подписки.")
    
print(f"Средняя продолжительность поездок у пользователей без подписки: {df_unsub['duration'].mean():.2f} мин.")
print(f"Средняя продолжительность поездок у пользователей с подпиской: {df_sub['duration'].mean():.2f} мин.")
print('p-значение:', results.pvalue)


# По результатам проведенного t-теста было установлено, что нет статистически значимой разницы в продолжительности поездок между пользователями с подпиской и пользователями без подписки. Это означает, что на основе имеющихся данных нельзя утверждать, что пользователи с подпиской проводят больше времени на поездках по сравнению с пользователями без подписки.

# Нулевая гипотеза (H0): Среднее расстояние поездок пользователей равно 3130 метров.
# 
# Альтернативная гипотеза (H1): Подписчики в среднем проезжают расстояние больше 3130 метров.

# In[52]:


results = stats.ttest_1samp(df_sub['distance'], 3130, alternative = 'greater')

if results.pvalue < alpha:
    print("Среднее расстояние пользователей с подпиской не превышает 3130 метров (статистически значимо).")
else:
    print("Среднее расстояние пользователей с подпиской превышает 3130 метров (статистически значимо).")
    
print(f"Среднее расстояние: {df_sub['distance'].mean():.2f}")
print(f"p-value: {results.pvalue}")


# Полученное p-value больше обычно используемого уровня значимости 0.05. Таким образом, на основе этой выборки данных у нас нет достаточных доказательств для отвержения нулевой гипотезы. Это означает, что мы не можем утверждать, что среднее расстояние пользователей с подпиской превышает 3130 метров с высокой степенью статистической значимости.
# 
# Вывод: На основе проведенного теста, нет достаточных статистических доказательств для утверждения, что среднее расстояние пользователей с подпиской за одну поездку превышает 3130 метров.

# In[53]:


aggregated_df = aggregated_df.merge(data_users, on='user_id', how='outer')
aggregated_df = aggregated_df.drop(columns=['name', 'age', 'city'])
aggregated_df


# In[54]:


revenue_subscription = aggregated_df[aggregated_df['subscription_type'] == 'ultra']
revenue_no_subscription = aggregated_df[aggregated_df['subscription_type'] == 'free']

for month in aggregated_df['month'].unique():
    revenue_month_subscription = revenue_subscription[revenue_subscription['month'] == month]['revenue']
    revenue_month_no_subscription = revenue_no_subscription[revenue_no_subscription['month'] == month]['revenue']

    t_stat, p_value = stats.ttest_ind(revenue_month_subscription, revenue_month_no_subscription)
    
    if p_value < alpha:
        print(f"В месяце {month}: Средняя выручка от пользователей с подпиской статистически значимо выше.")
    else:
        print(f"В месяце {month}: Нет статистически значимой разницы в средней выручке между группами.")


# Исходя из описания, для всех месяцев результаты t-теста показывают, что средняя выручка от пользователей с подпиской статистически значимо выше, чем средняя выручка от пользователей без подписки. В таком случае, можно сделать вывод о том, что во всех месяцах, указанных в результатах теста, выручка от пользователей с подпиской статистически значимо выше.
# 
# Это может быть полезной информацией для маркетинговых решений, так как она подтверждает, что пользователи с подпиской приносят больше выручки в сравнении с пользователями без подписки в большинстве месяцев.

# Для проверки гипотезы о том, что обновление серверов снизило количество обращений в техподдержку, мы можем использовать статистический тест на сравнение двух средних значений (до обновления и после обновления) и определить, есть ли статистически значимая разница между ними. Будем использовать t-тест для независимых выборок. Будем проверять одностороннюю гипотезу.

# #### Шаг 7. Распределения

# In[55]:


p = 0.1

# Желаемая вероятность не выполнить план (5%)
disired_probability = 0.05

# Начальное количество пользователей, которым предложена акция
n = 1

# Инициализируем текущую вероятность
current_probability = 0

# Поиск минимального n
while current_probability < 1 - disired_probability:
    current_probability = 1 - stats.binom.cdf(99, n , p)
    n += 1
    
# Вывод минимального n
print(f"Минимальное количество промокодов: {n - 1}")

# Построение графика биномиального распределения
x = np.arange(1, n)
y = [1 - stats.binom.cdf(99, i, p) for i in x]

plt.plot(x, y, marker='o')
plt.xlabel('Количество промокодов')
plt.ylabel('Вероятность не выполнить план')
plt.title('Распределение вероятности не выполнить план')
plt.grid()
plt.show()


# In[56]:


# Общее количество уведомлений
n = 1000000  

# Вероятность успеха (открытие уведомления)
p = 0.4  

# Не более 399,5 тыс. пользователей
desired_users = 399500  

# Вычисление математического ожидания и дисперсии биномиального распределения
mu = n * p
variance = n * p * (1 - p)

# Стандартное отклонение
sigma = variance**0.5

# Значение z-оценки для не более 399,5 тыс. пользователей
z = (desired_users - mu) / sigma

# Используем функцию нормального распределения для расчета вероятности
probability = stats.norm.cdf(z)

print(f"Вероятность открытия не более {desired_users} пользователей: {probability:.4f}")

# Создание массива значений
x = np.arange(0, n+1)

# Построение биномиального распределения
binom_dist = stats.binom.pmf(x, n, p)

# Построение аппроксимации нормального распределения
norm_dist = stats.norm.pdf(x, mu, sigma)

plt.figure(figsize=(10, 6))
plt.plot(x, binom_dist, label='Биномиальное распределение', marker='o', linestyle='-')
plt.plot(x, norm_dist, label='Аппроксимация нормальным распределением', linestyle='--')
plt.xlabel('Количество открытых уведомлений')
plt.ylabel('Вероятность')
plt.title('График распределения открытых уведомлений')
plt.legend()
plt.grid()
plt.show()


# В ходе обработки данных о пользователях и их поездках в сервисе аренды самокатов GoFast были удалены дубликаты пользователей с одинаковыми ID.
# 
# В ходе анализа данных о пользователях и их поездках в сервисе аренды самокатов GoFast были выявлены следующие ключевые моменты:
# 
# Распределение пользователей по городам показало, что наибольшее количество пользователей находится в Екатеринбурге, а наименьшее - в Тюмени. Однако разница в количестве пользователей между этими городами незначительна.
# 
# Доля пользователей с подпиской 'ultra' составляет 45.57%, в то время как пользователей без подписки - 54.43%. Это говорит о том, что большинство пользователей предпочитают использовать сервис без подписки.
# 
# Возраст пользователей имеет нормальное распределение, что может быть полезным для определения целевой аудитории и стратегии маркетинга.
# 
# Среднее расстояние, которое пользователи преодолевают за одну поездку, составляет около 3 километров, а среднее время продолжительности поездки - около 18 минут. Эти данные могут быть полезными при оптимизации ресурсов и обслуживания клиентов.
# 
# Проведенный t-тест не показал статистически значимой разницы в продолжительности поездок между пользователями с подпиской и без подписки. Это означает, что на основе имеющихся данных нельзя утверждать, что пользователи с подпиской проводят больше времени на поездках по сравнению с пользователями без подписки.
# 
# Проверка гипотезы о среднем расстоянии поездок также не выявила статистически значимой разницы между пользователями с подпиской и без нее. Среднее расстояние поездок для обеих групп оказалось близким к 3 километрам.
# 
# Анализ выручки по месяцам показал, что во всех месяцах средняя выручка от пользователей с подпиской статистически значимо выше, чем от пользователей без подписки. Эти данные могут быть важными при разработке маркетинговых стратегий.
# 
# В целом, анализ данных позволяет сделать вывод, что пользователи с подпиской и без подписки в целом ведут себя схоже, и нет статистически значимых различий в продолжительности поездок и расстоянии между этими группами. Однако, с учетом разницы в средней выручке, бизнесу стоит обратить внимание на стимулирование пользователей к подписке, так как это может привести к увеличению доходов.
