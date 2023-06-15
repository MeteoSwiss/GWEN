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

    # Create the plot
    fig, ax = plt.subplots()
    ax.plot(x, y1, label='Sine of x')
    ax.plot(x, y2, label='Cosine of x')

    # Add labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Simple Line Plot')

    # Add legends
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1, title='Legend 1: This is a very long legend title that will span multiple lines')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 0.8), ncol=1, title='Legend 2: This is another very long legend title that will also span multiple lines')

    # Show the plot
    plt.show()

plot_line()

test_plot = plot_line()