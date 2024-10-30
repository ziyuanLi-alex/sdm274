import numpy as np
import matplotlib.pyplot as plt


def complex_nonlin_func(x):
    return (
        np.sin(x) -
        np.sinc(3*x) +
        0.8 * np.exp(- x ** 2) +
        0.3 * np.exp( np.abs(x)**0.5 )
    )


if __name__ == "__main__":
    x_vals = np.linspace(-10, 10, 500)
    Y = complex_nonlin_func(x_vals)

    # Plot the function
    plt.plot(x_vals, Y)
    plt.show()
