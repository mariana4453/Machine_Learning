import numpy as np

# for building linar regression model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# plotting
from matplotlib import pyplot as plt

# reduce display precision on numpy array
np.set_printoptions(precision=2)

####################################################################
path = './01_data/LR_model_data.csv'
data = np.loadtxt(path, delimiter=',')

# splitting data inputs
x = data[:, 0]
y = data[:, 1]

# converting to 2D array
x = np.expand_dims(x, axis=1)
y = np.expand_dims(y, axis=1)

# plotting data
# plt.scatter(x, y, marker='x', color='red')

####################################################################
# splitting dataset: 60% - training set, 20% - validation, 20% - test
x_train, x_, y_train, y_ = train_test_split(x, y, test_size=0.2, random_state=1)

# splitting 40%: validation and test set
x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.5, random_state=1)

# plotting data sets distribution
# plt.scatter(x_train, y_train, marker='x', color='red')
# plt.scatter(x_cv, y_cv, color='blue', marker='v')
# plt.scatter(x_test, y_test, color='green', marker='*')
# plt.show()


# fitting data
scaler_linear = StandardScaler()
X_train_scaled = scaler_linear.fit_transform(x_train)

print(f"Mean of training set: {scaler_linear.mean_.squeeze():.2f}")
print(f"Standard deviation of training set: {scaler_linear.scale_.squeeze(): .2f}")

# training model
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# evaluating model
yhat = linear_model.predict(X_train_scaled)
print(f"training MSE (using sklearn function): {mean_squared_error(y_train, yhat) / 2 }")


# checking validation set
X_cv_scaled = scaler_linear.transform(x_cv)

print(f"Mean of CV set: {scaler_linear.mean_.squeeze():.2f}")
print(f"Standard deviation of CV set: {scaler_linear.scale_.squeeze(): .2f}")

yhat = linear_model.predict(X_cv_scaled)
print(f"training MSE of CV (using sklearn function): {mean_squared_error(y_cv, yhat) / 2 }")


# adding polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)

# Preview the first 5 elements of the new training set. Left column is `x` and right column is `x^2`
# Note: The `e+<number>` in the output denotes how many places the decimal point should
# be moved. For example, `3.24e+03` is equal to `3240`
X_train_mapped = poly.fit_transform(x_train)

scaler_poly = StandardScaler()
X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)

model = LinearRegression()
model.fit(X_train_mapped_scaled, y_train)

yhat = model.predict(X_train_mapped_scaled)
print(f"training MSE poly 2 (using sklearn function): {mean_squared_error(y_train, yhat) / 2}")

# checking on CV data
X_cv_mapped = poly.transform(x_cv)
X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)

yhat = model.predict(X_cv_mapped_scaled)
print(f"CV MSE poly 2 (using sklearn function): {mean_squared_error(y_cv, yhat) / 2 } \n")


# making loop for different poly degrees
train_mses = []
cv_mses = []
models = []
scalers = []

for degree in range(1, 11):
    # adding polynomial feature to training set
    poly = PolynomialFeatures(degree, include_bias=False)
    X_train_mapped = poly.fit_transform(x_train)

    # scale the training set
    scaler_poly = StandardScaler()
    X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)
    scalers.append(scaler_poly)

    # create and train model
    model = LinearRegression()
    model.fit(X_train_mapped_scaled, y_train)
    models.append(model)

    # compute training MSE
    ytah = model.predict(X_train_mapped_scaled)
    train_mse = np.round(mean_squared_error(y_train, ytah) / 2, 2)
    train_mses.append(train_mse)

    # adding poly feature to cv set
    poly = PolynomialFeatures(degree, include_bias=False)
    X_cv_mapped = poly.fit_transform(x_cv)
    X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)

    # compute cv MSE
    ytah = model.predict(X_cv_mapped_scaled)
    cv_mse = np.round(mean_squared_error(y_cv, ytah) / 2, 2)
    cv_mses.append(cv_mse)

# plotting mse of training and cv sets
# degrees = range(1, 11)
# plt.plot(degrees, train_mses, marker='x')
# plt.plot(degrees, cv_mses, marker='*')
# plt.show()

degree = np.argmin(cv_mses) + 1
print(f"Lowest CV MSE is found in the model with degree = {degree} \n")

# generalizing
poly = PolynomialFeatures(degree, include_bias=False)
X_test_mapped = poly.fit_transform(x_test)

X_test_mapped_scaled = scalers[degree - 1].transform(X_test_mapped)
yhat = models[degree - 1].predict(X_test_mapped_scaled)
test_mse = mean_squared_error(y_test, yhat) / 2

print(f"Training MSE: {train_mses[degree - 1]: .2f}")
print(f"Cross validation MSE: {cv_mses[degree - 1]} .2f")
print(f"Test MSE: {test_mse}")