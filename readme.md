# 🚗 Car Price Prediction with XGBoost

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-green?style=for-the-badge&logo=xgboost)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

Практическая работа по прогнозированию рыночной стоимости подержанных автомобилей на основе их технических характеристик и истории владения с использованием алгоритма градиентного бустинга **XGBoost Regressor**.

---

## 📌 Описание проекта

Цель проекта — построить модель машинного обучения для предсказания цены автомобиля (`selling_price`) на основе датасета из 30 000 записей. В ходе работы выполнен полный цикл Data Science проекта:

- 🧹 **Очистка данных**: обработка пропусков и удаление неинформативных текстовых столбцов (`name`).
- ⚙️ **Feature Engineering**: кодирование категориальных признаков с помощью **One-Hot Encoding** (`pd.get_dummies`).
- 🧪 **Эксперименты**: сравнение моделей с различной глубиной деревьев (`max_depth`), количеством итераций (`n_estimators`) и скоростью обучения (`learning_rate`).
- 📊 **Оценка качества**: расчёт метрик MAE, RMSE и R².
- 🔍 **Анализ важности признаков**: выявление факторов, наиболее сильно влияющих на стоимость авто.

---

## 📁 Структура проекта

```text
.
├── 📄 car_data.csv            # Исходный датасет (30 000 авто)
├── 📜 main.py                  # Основной скрипт обучения и визуализации
├── 📋 requirements.txt        # Список зависимостей Python
├── 📊 real_vs_predicted.png   # График: Реальная цена vs Предсказанная
├── 📈 feature_importance.png  # График: Топ-10 важных признаков
└── 📖 README.md               # Документация проекта