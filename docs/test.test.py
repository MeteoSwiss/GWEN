import matplotlib.pyplot as plt
import numpy as np


def plot_line():
    """
    Plots a simple line graph using Matplotlib.
    """
    # Generate some data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.tan(x)

    # Create the plot
    fig, ax = plt.subplots()
    ax.plot(x, y1, label='Sine')
    ax.plot(x, y2, label='Cosine')
    ax.plot(x, y3, label='Tangent')

    # Add labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Trigonometric Functions')

    # Add legends
    ax.legend(loc='upper left', bbox_to_anchor=(1, 0.8), ncol=1, title='Legend')

    # Show the plot
    plt.show()
