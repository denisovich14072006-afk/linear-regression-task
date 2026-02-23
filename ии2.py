import numpy as np
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 1. Аналитическое решение
def linear_regression_normal(X, y, add_intercept=True):
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if add_intercept:
        X = np.hstack([np.ones((X.shape[0], 1)), X])
    w = np.linalg.pinv(X.T @ X) @ X.T @ y
    return w.flatten()

# 2. Полный градиентный спуск (SSE, без деления на n)
def linear_regression_gd(X, y, lr=0.001, epochs=1000, add_intercept=True):
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if add_intercept:
        X = np.hstack([np.ones((X.shape[0], 1)), X])
    w = np.zeros((X.shape[1], 1))
    for _ in range(epochs):
        error = X @ w - y
        gradient = 2 * X.T @ error
        w -= lr * gradient
    return w.flatten()

# 3. Стохастический градиентный спуск (с мини-батчами)
def linear_regression_sgd(X, y, lr=0.01, epochs=100, batch_size=32, add_intercept=True):
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if add_intercept:
        X = np.hstack([np.ones((X.shape[0], 1)), X])
    w = np.zeros((X.shape[1], 1))
    n = X.shape[0]
    for _ in range(epochs):
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        for i in range(0, n, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            error = X_batch @ w - y_batch
            gradient = 2 * X_batch.T @ error
            w -= lr * gradient
    return w.flatten()

# Генерация данных
np.random.seed(42)
n, d = 200, 5
X = np.random.randn(n, d)
true_w = np.array([1.5, -2.0, 0.8, 3.1, -1.2])
y = X @ true_w + 0.2 * np.random.randn(n)

# Обучение наших моделей (без свободного члена для простоты)
w_normal = linear_regression_normal(X, y, add_intercept=False)
w_gd = linear_regression_gd(X, y, lr=0.001, epochs=2000, add_intercept=False)
w_sgd = linear_regression_sgd(X, y, lr=0.001, epochs=100, batch_size=32, add_intercept=False)

# sklearn LinearRegression
lr_sk = LinearRegression(fit_intercept=False).fit(X, y)
w_sk = lr_sk.coef_

# sklearn SGDRegressor (требует масштабирования)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
sgd_sk = SGDRegressor(fit_intercept=False, max_iter=1000, tol=1e-3, eta0=0.01, learning_rate='constant', random_state=42)
sgd_sk.fit(X_scaled, y)
w_sgd_sk = sgd_sk.coef_

# Сравнение
print("Истинные веса:          ", true_w)
print("Аналитика (наша):        ", w_normal)
print("GD (наша):               ", w_gd)
print("SGD (наша):              ", w_sgd)
print("sklearn LinearRegression:", w_sk)
print("sklearn SGDRegressor:    ", w_sgd_sk)

# Оценка MSE на тех же данных
y_pred_normal = X @ w_normal
y_pred_gd = X @ w_gd
y_pred_sgd = X @ w_sgd
y_pred_sk = X @ w_sk
y_pred_sgd_sk = X_scaled @ w_sgd_sk   # важно: X_scaled использовался для обучения

print("\nMSE:")
print(f"  Normal equation: {mean_squared_error(y, y_pred_normal):.6f}")
print(f"  GD:              {mean_squared_error(y, y_pred_gd):.6f}")
print(f"  SGD:             {mean_squared_error(y, y_pred_sgd):.6f}")
print(f"  sklearn LR:      {mean_squared_error(y, y_pred_sk):.6f}")
print(f"  sklearn SGD:     {mean_squared_error(y, y_pred_sgd_sk):.6f}")
