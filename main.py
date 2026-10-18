import numpy as np

from data import generate_data
from solver import normal_solver, lstsq_solver
from visualization import visualize

np.set_printoptions(precision=2)
np.random.seed(42)

x, A, y = generate_data()

lstsq_theta, lstsq_sse = lstsq_solver(A=A, y=y)
lstsq_mse = lstsq_sse / y.shape[0]
print("lstsq_theta:\n", lstsq_theta)
print("lstsq_sse:\n", lstsq_sse)
print("lstsq_mse:\n", lstsq_mse)

my_theta, my_sse = normal_solver(A=A, y=y)
if my_theta is not None:
    my_mse = my_sse / y.shape[0]
    print("my_theta:\n", my_theta)
    print("my_sse:\n", my_sse)
    print("my_mse:\n", my_mse)

    theta_is_same = np.allclose(my_theta, lstsq_theta)
    sse_is_same = np.allclose(my_sse, lstsq_sse)
    mse_is_same = np.allclose(my_mse, lstsq_mse)
    print("theta_is_same:", theta_is_same)
    print("sse_is_same:", sse_is_same)
    print("mse_is_same:", mse_is_same)
else:
    print("This normal_solver can not solve it.")

visualize(x=x, y=y, lstsq_theta=lstsq_theta, my_theta=my_theta)
