# This program constructs the interpolation Lagrange polynomial 𝐿𝑛(𝑥) for the function 𝑓(𝑥),
# and calculates the approximate values of the function at the specified points with an accuracy of 0.001.
# Also it plots the graph of the interpolation function 𝑦=𝐿𝑛(𝑥) on the available set of points.

import matplotlib.pyplot as plt
from numpy import arange

arr_x = []
arr_f = []

print(u"Enter first 4 given x\u1d62 along with their given f(x\u1d62)")

for i in range(4):
    arr_x.append(float(input(f"Enter x{i}: ")))
    arr_f.append(float(input(f"Enter f(x{i}): ")))

print()

n = int(input("Enter amount of the x\u1d62's with unknown f(x\u1d62) value: "))
arr_xad = []
for i in range(5, n+5):
    arr_xad.append(float(input(f"Enter x{i}: ")))

arr_L3 = []
for x in arr_xad:
    L3_1 = (((x-arr_x[1])*(x-arr_x[2])*(x-arr_x[3])*arr_f[0])/((arr_x[0]-arr_x[1])*(arr_x[0]-arr_x[2])*(arr_x[0]-arr_x[3])))
    L3_2 = (((x-arr_x[0])*(x-arr_x[2])*(x-arr_x[3])*arr_f[1])/((arr_x[1]-arr_x[0])*(arr_x[1]-arr_x[2])*(arr_x[1]-arr_x[3])))
    L3_3 = (((x-arr_x[0])*(x-arr_x[1])*(x-arr_x[3])*arr_f[2])/((arr_x[2]-arr_x[0])*(arr_x[2]-arr_x[1])*(arr_x[2]-arr_x[3])))
    L3_4 = (((x-arr_x[0])*(x-arr_x[1])*(x-arr_x[2])*arr_f[3])/((arr_x[3]-arr_x[0])*(arr_x[3]-arr_x[1])*(arr_x[3]-arr_x[2])))
    L3 = round((L3_1 + L3_2 + L3_3 + L3_4), 4)

    arr_L3.append(L3)

x = arr_x + arr_xad # [-3.0, -2.0, 0.0, 2.0, -4.0, -1.5, -1.0, 1.5]
y = arr_f + arr_L3 # [-22.0, -13.0, -7.0, 23.0, -43.0, -11.125, -10.0, 10.625]

func = zip(x, y)
func = sorted(func)

print('x =', x, '\n', 'y =', y)

plt.figure()

plt.plot(*zip(*func), c = 'k')
plt.scatter(*zip(*func), c = 'r')
plt.title('Graph of the interpolated function')
plt.xlabel('axis X')
plt.ylabel('axis Y')
plt.grid(True)

plt.show()