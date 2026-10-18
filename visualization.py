import numpy as np
import matplotlib.pyplot as plt

def visualize(x, y, lstsq_theta, my_theta):
    t0 = np.linspace(-2, 12, 100)
    t2 = t1 = t0

    plt.figure(figsize=(10, 10))
    plt.title("Data and Fitted Line")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)

    plt.scatter(x, y, marker='o', label='Raw data point')
    plt.plot(t0, (2 * t0 + 1), label='Ground truth(y=2*x+1)', color='blue')
    plt.plot(t1, (lstsq_theta[0]*t1+lstsq_theta[1]), label='lstsq_theta', color='green')
    if my_theta is not None:
        plt.plot(t2, (my_theta[0]*t2+my_theta[1]), label='my_theta', color='red')
    plt.legend(loc='upper left')

    plt.savefig("result.pdf")
    plt.show()