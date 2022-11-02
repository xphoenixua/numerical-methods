# This program solves the nonlinear algebraic equation f(x)=0
# Refinement of the roots is carried out by Newton's (tangent) metho
# and by the bisection method. Accuracy is 0.0001.

import sympy as sym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

infMin = -10**29
infPlus = 10**29

def graphPlot(y_parsed, xn, func_value):
    if input("\nPlot the graph? (y/n)\n")=='y':
        x_arr = np.arange(-20, 20, 0.5)
        y = np.array([0]*len(x_arr))
        for i in range(len(x_arr)):
            y[i] = y_parsed.subs(x, x_arr[i])
        plt.grid()
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.title('Graph of the function')
        plt.plot(x_arr, y, c='black', linewidth=1.5)
        plt.ylim(-3000, 3000)
        plt.scatter(xn, func_value, marker='o', c='red')
        plt.text(xn, func_value, f'({xn:.6f}, {func_value:.6f})')
        plt.show()

def funcSign(func_parsed, value, fl=0):
    if func_parsed.subs(x, value)<0:
        if fl==0:
            func_signs.append('-')
        return '-'
    elif func_parsed.subs(x, value)>0:
        if fl==0:
            func_signs.append('+')
        return '+'

def clearRow(xn, f, f_diff, difference):
    xn.pop()
    f.pop()
    f_diff.pop()
    difference.pop()

x, y = sym.symbols('x y')
PRECISION = float(input("Enter precision = "))
func = input("Input your function: ")
func_parsed = sym.sympify(func)
func_diff = sym.diff(func_parsed, x)
print("First derivative of the function:", func_diff)

if input("Try demo? (y/n)\n")=='y':
    func_roots = sym.solve(func_diff)
    func_roots = [float(x) for x in func_roots if x.is_real]
    print("\nExtremas of the function:", func_roots)
    func_signs = []
    funcSign(func_parsed, infMin)
    for root in func_roots:
        funcSign(func_parsed, root)
    funcSign(func_parsed, infPlus)
    func_roots = ['-∞'] + func_roots + ['+∞']
    df = pd.DataFrame([func_roots, func_signs], index=['roots', 'signs'])
    print(df)

    flag = func_signs[0]
    counter = 0
    index = 0
    roots_indeces = []
    for sign in func_signs:
        if sign=='+' and flag=='-':
            flag = '+'
            counter += 1
            roots_indeces.append(index-1)
            roots_indeces.append(index)
        elif sign=='-' and flag=='+':
            flag = '-'
            counter += 1
            roots_indeces.append(index-1)
            roots_indeces.append(index)
        index += 1

    flag = func_signs[0]
    counter = 0
    for sign in func_signs:
        if sign==flag:
            counter += 1
            if counter==2:
                func_roots.remove(func_roots[func_signs.index(sign)+1])
                func_signs.remove(sign)
        else:
            flag = sign
            counter = 0

    func_roots[0] = infMin
    func_roots[-1] = infPlus
    for number in range(int(func_roots[1]), int(func_roots[0]), -1):
        if funcSign(func_parsed, number, fl=1)==func_signs[0]:
            func_roots[0] = number
            break
    for number in range(int(func_roots[len(func_roots)-2]), func_roots[-1]):
        if funcSign(func_parsed, number, fl=1)==func_signs[len(func_roots)-1]:
            func_roots[-1] = number
            break

    for number in range(int(func_roots[1]), int(func_roots[0]), -1):
        if funcSign(func_parsed, number, fl=1)==func_signs[1]:
            func_roots[1] = number
    for number in range(int(func_roots[2]), int(func_roots[3]+1)):
        if funcSign(func_parsed, number, fl=1)==func_signs[3]:
            func_roots[2] = number-1
    
    print()
    print("Zooming...")
    df = pd.DataFrame([func_roots, func_signs], index=['roots', 'signs'])
    print(df)
    print()
    it = iter(func_signs)
    signs = list(zip(it, it))

else:
    func_roots = [0]*4
    for i in range(4):
        func_roots[i] = float(input("Enter extremas in ascending order: "))

roots = []
for i in range(0, len(func_roots), 2):
    roots.append(func_roots[i:i+2])

i = 1
for interval in roots:
    print(f"x{i} є {interval}")
    i += 1

ex = 'y'
while ex!='n':
    print("Which root to find?")
    choice_root = int(input())-1
    print("Find root with: \n1. Newthon Method \n2. Bisections Method")
    choice_method = input()
    print()

    if choice_method=='1':
        value_func = func_parsed.subs(x, roots[choice_root][0])
        func_2diff = sym.diff(func_diff, x)
        value_2diff = func_2diff.subs(x, roots[choice_root][0])
        print("Second derivative of the function:", func_2diff)
        if (value_func*value_2diff>0):
            nwtFlag = 0
        else:
            nwtFlag = 1

        xn = [roots[choice_root][nwtFlag]]
        f = [value_func]
        f_diff = [func_diff.subs(x, xn[0])]
        difference = [1]
        i = 0
        while f[i]>PRECISION or difference[i]>PRECISION:
            i += 1
            xn.append(xn[i-1]-float(f[i-1]/f_diff[i-1]))
            f.append(func_parsed.subs(x, xn[i]))
            f_diff.append(func_diff.subs(x, xn[i]))
            difference.append(abs(xn[i]-xn[i-1]))
        clearRow(xn, f, f_diff, difference)
        difference[0] = None
        df_1 = pd.DataFrame(data=[xn, f, f_diff, difference], index=['xn', 'f(xn)', "f'(xn)", '|xn-xn-1|'])
        df_1 = df_1.transpose()
        print(df_1)
        print()
        print(f"x{choice_root+1} ≈ {xn[-1]:.6f}")

    elif choice_method=='2':
        an = [roots[choice_root][0]]
        bn = [roots[choice_root][1]]
        xn = [float((an[0]+bn[0])/2)]
        f = [func_parsed.subs(x, xn[0])]
        difference = [abs(an[0]-bn[0])]
        i = 0
        while abs(f[i])<PRECISION or difference[i]>2*PRECISION:
            i += 1
            if f[i-1]<0:
                an.append(xn[i-1])
                bn.append(bn[i-1])
            elif f[i-1]>0:
                bn.append(xn[i-1])
                an.append(an[i-1])
            xn.append(float((an[i]+bn[i])/2))
            f.append(func_parsed.subs(x, xn[i]))
            difference.append(abs(an[i]-bn[i]))
        df_2 = pd.DataFrame(data=[an, bn, xn, f, difference], index=['an', 'bn', 'xn', 'f(xn)', '|an-bn|'])
        df_2 = df_2.transpose()
        print(df_2)
        print()
        print(f"x{choice_root+1} ≈ {xn[-1]:.6f}")
    
    graphPlot(func_parsed, xn[-1], f[-1])
    ex = input("\nFind another root? (y/n)\n")
