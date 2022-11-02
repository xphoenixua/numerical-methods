# This program calculates the value of the defined integrals
# by the method of rectangles, by Simpson's method, the trapezium method
# with an accuracy of 0.0001

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from py_expression_eval import Parser

parser = Parser()
global PRECISION
PRECISION = 5

def hCalc(n,lb,rb):
    if rb>lb and n>0:
        h = (rb-lb)/n
        return h

def xyTable(func,x,h,n,half,task,a=0,b=0):
    if half==1:
        x = (x+h)[:-1]
        n-=1
    y = [0]*(n+1)
    for i in range(len(x)):
        if task==2:
            y[i] = parser.parse(func).evaluate({'x': x[i]})
        else:
            y[i] = parser.parse(func).evaluate({'x': x[i],'a': a, 'b': b})
    df = pd.DataFrame([x,y], index=['x','y'])
    print(df)
    return y

def leftRect(y,h):
    integral_val = h*(np.sum(y[:-1]))
    return integral_val

def rightRect(y,h):
    integral_val = h*(np.sum(y[1:]))
    return integral_val

def middleRect(y,h):
    integral_val = (h*2)*(np.sum(y))
    return integral_val

def simpson(y,h):
    y_even = 0
    y_odd = 0
    for i in range(1,len(y)-1):
        if i%2!=0:
            y_odd+=y[i]
        else:
            y_even+=y[i]
    integral_val = (h/3)*(y[0]+y[-1]+4*y_odd+2*y_even)
    return integral_val

def trapezoid(y,h):
    integral_val = h*((y[0]+y[-1])/2+sum(y[1:-1]))
    return integral_val

def integralValues(y,h):
    intval_lrect = round(leftRect(y, h), PRECISION)
    intval_rrect = round(rightRect(y, h), PRECISION)
    intval_simp = round(simpson(y, h), PRECISION)
    intval_trap = round(trapezoid(y, h), PRECISION)

    print("Left rectangles' method: ", intval_lrect)
    print("Right rectangles' method: ", intval_rrect)
    print("Simpson's method: ", intval_simp)
    print("Trapezoids' method: ", intval_trap)

def graphPlot(x,y):
    if input("\nPlot the graph? (y/n)\n")=='y':
        plt.figure()
        plt.grid()
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.title('Graph of the function')
        f = interp1d(x, y)
        plt.plot(x, f(x), c='black', linewidth=1.5)
        plt.scatter(x, y, marker='o', c='red')
        plt.show()

flag = 0
choice = 'y'
while choice!='n':
    while flag!=1:
        print()
        task = int(input("Enter the task number: "))
        lb = float(input("Lower bound of the integral: "))
        rb = float(input("Upper bound of the integral: "))
        n = int(input("Number of the division segments: "))  
        h = hCalc(n, lb, rb)
        x = np.arange(lb,rb+h,h)[:-1]

        if task==1:
            flag = 1
            print("Integral has the form of ∫1/√(ax+b)")
            func = "1/(a*x+b)^0.5"
            a = float(input("a = "))
            b = float(input("b = "))
            print()

            y = xyTable(func, x, h, n, 0, 1, a, b)
            integralValues(y,h)
            print()
            y_half = xyTable(func, x, h/2, n, 1, 1, a, b)
            intval_hrect = round(middleRect(y_half, h/2), PRECISION)
            print("Middle rectangles' method: ", intval_hrect)

            graphPlot(x,y)

        elif task==2:
            flag = 1
            func = input("Enter the function: f(x) = ")
            print()

            y = xyTable(func, x, h, n, 0, 2)
            integralValues(y,h)
            print()
            y_half = xyTable(func, x, h/2, n, 1, 2)
            intval_hrect = round(middleRect(y_half, h/2), PRECISION)
            print("Middle rectangles' method: ", intval_hrect)

            graphPlot(x,y)

        elif task==3:
            flag = 1
            print("Integral has the form of ∫1/√(ax²+b)")
            func = "1/(a*(x^2)+b)^0.5"
            a = float(input("a = "))
            b = float(input("b = "))
            print()

            y = xyTable(func, x, h, n, 0, 3, a, b)
            integralValues(y,h)
            print()
            y_half = xyTable(func, x, h/2, n, 1, 3, a, b)
            intval_hrect = round(middleRect(y_half, h/2), PRECISION)
            print("Middle rectangles' method: ", intval_hrect)

            graphPlot(x,y)

    print()
    flag = 0
    choice = input("Complete another task? (y/n)\n")
