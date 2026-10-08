import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

def main():
    print("==========================================")
    print("1. ЗАГРУЗКА И ИССЛЕДОВАНИЕ ДАТАСЕТА")
    print("==========================================")
    df = pd.read_csv("car_data.csv")

    print(f"Размер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")
    print("\nПервые 5 строк:")
    print(df.head())

    print("\nИнформация о типах данных и пропусках:")
    print(df.info())

    print("\nКоличество пропусков по столбцам:")
    print(df.isnull().sum())

    print("\n==========================================")
    print("2. ПОДГОТОВКА И ОЧИСТКА ДАННЫХ")
    print("==========================================")
    # Удаляем столбец name и строки с пропусками
    df_clean = df.drop("name", axis=1).dropna()
    print(f"Размер после очистки: {df_clean.shape}")

    # Кодирование категориальных признаков
    df_encoded = pd.get_dummies(df_clean, drop_first=True)

    X = df_encoded.drop("selling_price", axis=1)
    y = df_encoded["selling_price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n==========================================")
    print("3. ЭКСПЕРИМЕНТ №1 (Задание 4)")
    print("n_estimators=100, max_depth=3, learning_rate=0.1")
    print("==========================================")
    model1 = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    model1.fit(X_train, y_train)
    pred1 = model1.predict(X_test)

    mae1 = mean_absolute_error(y_test, pred1)
    rmse1 = root_mean_squared_error(y_test, pred1)
    r2_1 = r2_score(y_test, pred1)
    print(f"MAE: {mae1:.2f} | RMSE: {rmse1:.2f} | R²: {r2_1:.4f}")

    print("\n==========================================")
    print("4. ЭКСПЕРИМЕНТ №2 (Задание 5)")
    print("n_estimators=300, max_depth=6, learning_rate=0.05")
    print("==========================================")
    model2 = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
    model2.fit(X_train, y_train)
    pred2 = model2.predict(X_test)

    mae2 = mean_absolute_error(y_test, pred2)
    rmse2 = root_mean_squared_error(y_test, pred2)
    r2_2 = r2_score(y_test, pred2)
    print(f"MAE: {mae2:.2f} | RMSE: {rmse2:.2f} | R²: {r2_2:.4f}")

    print("\n==========================================")
    print("5. ЭКСПЕРИМЕНТ №3 (Собственный вариант)")
    print("n_estimators=500, max_depth=8, learning_rate=0.03")
    print("==========================================")
    model3 = XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.03, random_state=42)
    model3.fit(X_train, y_train)
    pred3 = model3.predict(X_test)

    mae3 = mean_absolute_error(y_test, pred3)
    rmse3 = root_mean_squared_error(y_test, pred3)
    r2_3 = r2_score(y_test, pred3)
    print(f"MAE: {mae3:.2f} | RMSE: {rmse3:.2f} | R²: {r2_3:.4f}")

    print("\n==========================================")
    print("6. ВАЖНОСТЬ ПРИЗНАКОВ И ГРАФИКИ")
    print("==========================================")
    importance = pd.Series(model2.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("Топ-10 важных признаков:")
    print(importance.head(10))

    # Построение и сохранение графиков
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, pred2, alpha=0.4, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Реальная цена")
    plt.ylabel("Предсказанная цена")
    plt.title("Реальные vs Предсказанные цены (Модель №2)")
    plt.tight_layout()
    plt.savefig("real_vs_predicted.png")

    plt.figure(figsize=(9, 5))
    importance.head(10).plot(kind="bar", color="skyblue")
    plt.title("Топ-10 наиболее важных признаков")
    plt.ylabel("Важность")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    print("\nГрафики успешно сохранены в файлы png!")

if __name__ == "__main__":
    main()