import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# np.array(): converts a Python list into a NumPy array so we can do maths on every element at once
# Raw data
years = np.array([1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996])
x = years

y = np.array([12400, 10900, 10000, 1050, 9500, 8900, 8000, 7800, 7600, 7200])

# plt.figure(): creates a new blank figure canvas BEFORE plotting anything
# figsize=(8,5): width=8 inches, height=5 inches — controls the output image size
plt.figure(figsize=(8, 5))

# scatter(): draws individual dots (one per data point)
# color='blue': all dots are blue
# label='Data Points': text shown in the legend
plt.scatter(x, y, color='blue', label='Data Points')
plt.xlabel('Years since 1987 (x)')
plt.ylabel('Number of Insured Persons (y)')
plt.title('Insured Persons vs. Year')
# This forces the x-axis to display a tick mark for every value in your 'x' array
plt.xticks(x)
plt.legend()  # Draw the legend box using the label= strings defined in scatter/plot calls
plt.grid(True) # Add a background grid to make it easier to read values
plt.show()     # Render and display the figure

def build_design_matrix(x, degree):
    """
    Build Vandermonde design matrix.
    Column j contains x^j  (j = 0, 1, …, degree).
    """
    n=len(x)
    X_design=np.zeros((n,degree+1))

    for j in range(degree+1) :
        X_design[:, j] = x ** j
    return X_design

# np.linspace(start, stop, num): 200 evenly spaced values between 0 and 9
x_plot = np.linspace(x.min(), x.max(), 200)

models = {}
for degree in [1, 2, 3]:
    X_design = build_design_matrix(x, degree)

# ── To get the least squares and Get the Weights , @ is Matrix Multiplication
    # w = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y
    w = np.linalg.lstsq(X_design, y, rcond=None)[0]
    y_pred = X_design @ w

    SSE = np.sum((y - y_pred) ** 2)
    SS_yy = np.sum((y - np.mean(y)) ** 2)
    R_2 = 1 - SSE / SS_yy
    models[degree] = {'w': w, 'R_2': R_2}
    print(f"Degree {degree}: weights = {w}")
    print(f"  R² = {R_2:.4f}\n")

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='black', zorder=5, label='Data')
colors = {1: 'blue', 2: 'green', 3: 'red'}   
labels = {1: 'Linear', 2: 'Quadratic', 3: 'Cubic'}

for deg, data in models.items():
    # models.items(): iterates as (key=degree, value=dict with 'w' and 'r2')
    X_plot_mat = build_design_matrix(x_plot, deg)

    y_curve = X_plot_mat @ data['w']

    plt.plot(x_plot, y_curve, color=colors[deg],
             label=f"{labels[deg]} (R²={data['R_2']:.3f})")

plt.xlabel('x (years since 1987)')
plt.ylabel('y (insured persons)')
plt.title('Regression Models — Original Data')
plt.xticks(x)
plt.legend()
plt.grid(True)
plt.show()

mask = y != 1050
# Boolean mask: True for every element of y that is NOT equal to 1050
# Result: [True, True, True, False, True, True, True, True, True, True]

x_clean = x[mask]   # Boolean indexing: keeps only elements where mask=True
y_clean = y[mask]

models_clean = {}
for degree in [1, 2, 3]:
    X_design = build_design_matrix(x_clean, degree)

# ── To get the least squares and Get the Weights , @ is Matrix Multiplication
    #w = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y_clean
    w = np.linalg.lstsq(X_design, y_clean, rcond=None)[0]
    y_pred = X_design @ w

    SSE = np.sum((y_clean - y_pred) ** 2)
    SS_yy = np.sum((y_clean - np.mean(y_clean)) ** 2)
    R_2 = 1 - SSE / SS_yy
    models_clean[degree] = {'w': w, 'R_2': R_2}
    print(f"Degree {degree}: weights = {w}")
    print(f"  R² = {R_2:.4f}\n")

plt.figure(figsize=(10, 6))
plt.scatter(x_clean, y_clean, color='black', zorder=5, label='Data')
colors = {1: 'blue', 2: 'green', 3: 'red'}   
labels = {1: 'Linear', 2: 'Quadratic', 3: 'Cubic'}

for deg, data in models_clean.items():
    # models_clean.items(): iterates as (key=degree, value=dict with 'w' and 'r2')
    X_plot_mat = build_design_matrix(x_plot, deg)

    y_curve = X_plot_mat @ data['w']

    plt.plot(x_plot, y_curve, color=colors[deg],
             label=f"{labels[deg]} (R²={data['R_2']:.3f})")

plt.xlabel('x (years since 1987)')
plt.ylabel('y (insured persons)')
plt.title('Regression Models — Cleaned Data')
plt.xticks(x)
plt.legend()
plt.grid(True)
plt.show()

best_degree = 2
best_w = models_clean[best_degree]['w'] 

X_pred = build_design_matrix(np.array([1997]), best_degree)

prediction_1997 = (X_pred @ best_w)[0]

print(f"Predicted insured persons in 1997: {prediction_1997:.0f}")

