# This program solves the Cauchy problem
# for an ordinary differential equation of the first order
# on a segment with a step h=0.1 with the Euler's method.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from py_expression_eval import Parser

parser = Parser()
global PRECISION
PRECISION = 4

def xyTable(func, x, y, h, method):
    for i in range(len(x)-1):
        fi = parser.parse(func).evaluate({'x': x[i], 'y': y[i]})

        if method=='1':
            y_next = y[i] + h * fi

        if method=='2':
            y_new = y[i] + h * fi
            fi_new = parser.parse(func).evaluate({'x': x[i+1], 'y': y_new})
            y_next = y[i] + (h/2) * (fi + fi_new)
        
        y = np.append(y, y_next)

    if method=='2':
        x = x[:-1]
        y = y[:-1]
    df = pd.DataFrame([x,y], index=['x','y'])
    print(df)
    return x, y

def graphPlot(x, y, method):
    plt.figure()

    plt.grid()
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Graph of the function')
    plt.plot(x, y, c='black', linewidth=1.5)

    plt.scatter(x, y, marker='o', c='red')
    for i in range(len(x)):
        xi = np.round(x[i], PRECISION)
        yi = np.round(y[i], PRECISION)
        plt.annotate(f"({xi}; {yi})", (xi-0.01, yi-0.15))

    plt.show()

choice = 'y'
while choice!='n':
    print()

    lb = float(input("Lower bound of the interval: "))
    rb = float(input("Upper bound of the interval: "))
    h = float(input("Enter step: "))
    x0 = float(input("Enter x0: "))
    y0 = float(input("Enter y0: "))
    x = np.arange(lb, rb+h, h)
    y = np.array([y0])

    print("Enter first order different equation")
    func = input("y'=")
    print()

    print("1. Euler method\n2. Euler-Cauchy method\n1 or 2?")
    method = input()
    print()

    x, y = xyTable(func, x, y, h, method)
    print()

    if input("\nPlot the graph? (y/n)\n")=='y':
        graphPlot(x, y, method)
    print()

    choice = input("Continue? (y/n)\n")
