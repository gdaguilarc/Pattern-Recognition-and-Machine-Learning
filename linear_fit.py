import matplotlib.pyplot as plt
import tkinter as tk
import numpy as np

# Example values
# x = [-1, 12, 5, 4, 8]
# y = [0, 13, -4, 4, 0]


def fit_line():
    x = list(map(int, input_x.get().split(',')))
    y = list(map(int, input_y.get().split(',')))
    print(tabulation(linfit(x, y)))
    line = tabulation(linfit(x, y))
    plt.plot(x, y, 'ro')
    plt.plot(line, 'b')
    plt.show()


def linfit(x, y):
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(np.dot(x, y))
    sum_squaredx = np.sum(np.square(x))
    a = (((n*sum_xy)-(sum_x*sum_y))/((n*sum_squaredx) - sum_x*sum_x))
    b = ((sum_y - (a*sum_x)) / n)
    return [a, b]


def tabulation(variables):
    a = variables[0]
    b = variables[1]
    result = []
    for i in range(0, 20):
        result.append(a*i + b)
    return result


master = tk.Tk()
tk.Label(master, text="Xs array").grid(row=0)
tk.Label(master, text="Ys array").grid(row=1)
input_x = tk.Entry(master)
input_y = tk.Entry(master)
input_x.grid(row=0, column=1)
input_y.grid(row=1, column=1)
tk.Button(master, text="Compute", command=fit_line).grid(row=3, column=0)
tk.Button(master, text="Quit", command=master.quit).grid(row=3, column=1)
master.mainloop()
tk.mainloop()
