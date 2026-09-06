import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Raw Data: [Temperature, Insulation]
X = np.array([
    [40, 4], [27, 4], [40, 10], [73, 6], [65, 7], 
    [65, 40], [10, 6], [9, 10], [24, 10], [65, 4], 
    [66, 10], [41, 6], [22, 4], [40, 4], [60, 10]
])
y = np.array([270, 362, 162, 45, 91, 233, 372, 305, 234, 122, 25, 210, 450, 325, 52])

fig = plt.figure(figsize=(8, 6))
ax0 = fig.add_subplot(111, projection='3d')
ax0.scatter(X[:, 0], X[:, 1], y, color='blue', s=60, label='Raw Data Points')
ax0.set_title("Initial 3D Scatter Plot")
ax0.set_xlabel('Temperature (F)')
ax0.set_ylabel('Insulation (in)')
ax0.set_zlabel('Oil Usage')
plt.legend()
plt.show()

def build_multi_design_matrix(X_data, degree):
    """
    Builds a Vandermonde-style matrix for two variables.
    Degree 1: [1, x1, x2]
    Degree 2: [1, x1, x2, x1^2, x2^2, x1*x2]
    """
    x1 = X_data[:, 0]
    x2 = X_data[:, 1]
    n = len(X_data)
    
    if degree == 1:
        # Columns: Constant, Temp, Insulation
        return np.column_stack([np.ones(n), x1, x2])
    elif degree == 2:
        # Columns: Constant, Temp, Insulation, Temp^2, Insul^2, Temp*Insul
        return np.column_stack([np.ones(n), x1, x2, x1**2, x2**2, x1*x2])
    
def solve_regression(X_data, y_data):
    results = {}
    for deg in [1, 2]:
        X_design = build_multi_design_matrix(X_data, deg)
        
        # weights = (X^T @ X)^-1 @ X^T @ y (handled by lstsq for stability)
        w = np.linalg.lstsq(X_design, y_data, rcond=None)[0]
        y_pred = X_design @ w
        
        # Calculate R^2 manually
        sse = np.sum((y_data - y_pred)**2)
        ss_yy = np.sum((y_data - np.mean(y_data))**2)
        r2 = 1 - (sse / ss_yy)
        
        results[deg] = {'w': w, 'r2': r2}
        name = "Linear" if deg == 1 else "Quadratic"
        print(f"{name} R²: {r2:.4f}")
    return results

# Part A: Original Data
results_a = solve_regression(X, y)

# Part B: Cleaned Data (Remove index 5)
mask = np.ones(len(y), dtype=bool)
mask[5] = False
X_clean, y_clean = X[mask], y[mask]
results_b = solve_regression(X_clean, y_clean)

# Visualization
fig = plt.figure(figsize=(12, 5))

def add_subplot(fig, pos, X_d, y_d, res, title):
    ax = fig.add_subplot(pos, projection='3d')
    ax.scatter(X_d[:, 0], X_d[:, 1], y_d, color='black', s=40, zorder=10)
    
    # Create the prediction grid
    x1_range = np.linspace(X_d[:, 0].min(), X_d[:, 0].max(), 20)
    x2_range = np.linspace(X_d[:, 1].min(), X_d[:, 1].max(), 20)
    m1, m2 = np.meshgrid(x1_range, x2_range)
    grid_coords = np.column_stack([m1.ravel(), m2.ravel()])
    
    # Linear Surface
    X_grid_lin = build_multi_design_matrix(grid_coords, 1)
    z_lin = (X_grid_lin @ res[1]['w']).reshape(m1.shape)
    ax.plot_surface(m1, m2, z_lin, alpha=0.2, color='blue')
    
    # Quadratic Surface
    X_grid_quad = build_multi_design_matrix(grid_coords, 2)
    z_quad = (X_grid_quad @ res[2]['w']).reshape(m1.shape)
    ax.plot_surface(m1, m2, z_quad, alpha=0.5, cmap='YlGn')
    
    ax.set_title(f"{title}\nLin R2:{res[1]['r2']:.3f} | Quad R2:{res[2]['r2']:.3f}")
    ax.set_xlabel('Temp'); ax.set_ylabel('Insul'); ax.set_zlabel('Oil')

add_subplot(fig, 121, X, y, results_a, "Original (Part A)")
add_subplot(fig, 122, X_clean, y_clean, results_b, "Cleaned (Part B)")
plt.tight_layout()
plt.show()

# def plot_models(X_d, y_d, res, main_title):
#     fig = plt.figure(figsize=(16, 7))
    
#     # Grid for surfaces
#     x1_range = np.linspace(X_d[:, 0].min(), X_d[:, 0].max(), 20)
#     x2_range = np.linspace(X_d[:, 1].min(), X_d[:, 1].max(), 20)
#     m1, m2 = np.meshgrid(x1_range, x2_range)
#     grid_coords = np.column_stack([m1.ravel(), m2.ravel()])

#     # Subplot 1: Linear
#     ax1 = fig.add_subplot(121, projection='3d')
#     ax1.scatter(X_d[:, 0], X_d[:, 1], y_d, color='black', s=40, zorder=5)
#     z_lin = (build_multi_design_matrix(grid_coords, 1) @ res[1]['w']).reshape(m1.shape)
#     ax1.plot_surface(m1, m2, z_lin, alpha=0.6, cmap='Blues')
#     ax1.set_title(f"Linear Fit (R²: {res[1]['r2']:.4f})")
#     ax1.set_xlabel('Temp'); ax1.set_ylabel('Insul'); ax1.set_zlabel('Oil')

#     # Subplot 2: Quadratic
#     ax2 = fig.add_subplot(122, projection='3d')
#     ax2.scatter(X_d[:, 0], X_d[:, 1], y_d, color='black', s=40, zorder=5)
#     z_quad = (build_multi_design_matrix(grid_coords, 2) @ res[2]['w']).reshape(m1.shape)
#     ax2.plot_surface(m1, m2, z_quad, alpha=0.6, cmap='YlGn')
#     ax2.set_title(f"Quadratic Fit (R²: {res[2]['r2']:.4f})")
#     ax2.set_xlabel('Temp'); ax2.set_ylabel('Insul'); ax2.set_zlabel('Oil')

#     fig.suptitle(main_title, fontsize=16)
#     plt.show()

# plot_models(X, y, results_a, "Part A: Original Data")
# plot_models(X_clean, y_clean, results_b, "Part B: Cleaned Data")

# Part C: Prediction
# Input: 15 F, 5 inches
x_new = np.array([[15, 5]])
X_new_design = build_multi_design_matrix(x_new, 2)

# Prediction = X_design @ weights
prediction = (X_new_design @ results_b[2]['w'])[0]

print(f"\nPart C Prediction (15F, 5in): {prediction:.2f} gallons")