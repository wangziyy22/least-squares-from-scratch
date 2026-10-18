import numpy as np
import matplotlib.pyplot as plt


def visualize(x, y, lstsq_theta, my_theta):
    t = np.linspace(-2, 12, 100)

    plt.figure(figsize=(10, 10))
    plt.title("Data and Fitted Line")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)

    plt.scatter(x, y, marker="o", label="Raw data points")
    plt.plot(
        t, (2 * t + 1),
        label="Ground truth(y=2*x+1)", color="blue"
        )
    plt.plot(
        t, (lstsq_theta[0] * t + lstsq_theta[1]),
        label="lstsq_theta", color="green"
        )
    if my_theta is not None:
        plt.plot(
            t, (my_theta[0] * t + my_theta[1]),
            label="my_theta", color="red"
            )
    plt.legend(loc="upper left")

    plt.savefig("result.pdf")
    plt.show()
