# This program uses Newton's interpolation formulas with an accuracy of 0.001
# to find the values of the first and second derivatives

import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

precision = 3

def interpPhase(x_custom, h, mean):
    if x_custom <= mean:
        flag = 1
    else:
        flag = 2
    
    for i in range(len(xi)):
        if xi[i] > x_custom:
            if flag == 1:
                closest_supp = xi[i-1]
                index_supp = i-1
                break
            elif flag == 2:
                closest_supp = xi[i]
                index_supp = i
                break

    q = (x_custom - closest_supp) / h
    return [flag, index_supp, q]

def newtonFormula(x_data):
    if x_data[0] == 1:
        x_deriv1, x_deriv2 = newton1Derivative(x_data[1], x_data[2], df_1)
        x_data.append(x_deriv1)
        x_data.append(x_deriv2)
    elif x_data[0] == 2:
        x_deriv1, x_deriv2 = newton2Derivative(x_data[1], x_data[2], df_1)
        x_data.append(x_deriv1)
        x_data.append(x_deriv2)

    return x_data

def newton1Derivative(index_supp, q, df):
    deriv1_y = (1/h) * (df['Δy_i'][index_supp] + ((2*q - 1) / 2) * df['Δ2y_i'][index_supp] 
    + ((3*(q**2) - 6*q + 2) / 6) * df['Δ3y_i'][index_supp]
    + ((2*(q**3) - 9*(q**2) + 11*q - 3) / 12) * df['Δ4y_i'][index_supp])

    deriv2_y = (1 / (h**2)) * (df['Δ2y_i'][index_supp] + (q - 1) * df['Δ3y_i'][index_supp] 
    + ((6*(q**2) - 18*q + 11) / 12) * df['Δ4y_i'][index_supp])

    return (round(deriv1_y, precision), round(deriv2_y, precision))

def newton2Derivative(index_supp, q, df):
    deriv1_y = (1/h) * (df['Δy_i'][index_supp-1] + ((2*q + 1) / 2) * df['Δ2y_i'][index_supp-2] 
    + ((3*(q**2) + 6*q + 2) / 6) * df['Δ3y_i'][index_supp-3]
    + ((2*(q**3) + 9*(q**2) + 11*q + 3) / 12) * df['Δ4y_i'][index_supp-4])

    deriv2_y = (1 / (h**2)) * (df['Δ2y_i'][index_supp-2] + (q + 1) * df['Δ3y_i'][index_supp-3] 
    + ((6*(q**2) + 18*q + 11) / 12) * df['Δ4y_i'][index_supp-4])

    return (round(deriv1_y, precision), round(deriv2_y, precision))

n = int(input("Enter your personal number in the group's list: "))
if n%2 != 0:
    xi = [2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6]
    yi = [3.526, 3.782, 3.945, 4.043, 4.104, 4.155, 4.222, 4.331, 4.507, 4.775, 5.159, 5.683]
    x1_custom = 2.4 + 0.05 * n
    print("x_1:", x1_custom)
    x2_custom = 4.04 - 0.04 * n
    print("x_2:", x2_custom)
else:
    xi = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
    yi = [10.517, 10.193, 9.807, 9.387, 8.977, 8.637, 8.442, 8.482, 8.862, 9.701, 11.132, 13.302]
    x1_custom = 1.6 + 0.08 * n
    print("x_1:", x1_custom)
    x2_custom = 6.3 - 0.12 * n
    print("x_2:", x2_custom)

labels = ['i', 'x_i', 'y_i', 'Δy_i', 'Δ2y_i', 'Δ3y_i', 'Δ4y_i', 'Δ5y_i', 'Δ6y_i',
'Δ7y_i', 'Δ8y_i', 'Δ9y_i', 'Δ10y_i', 'Δ11y_i']
table_finite_diff = [xi, yi]
for i in range(1,12):
    diy = []
    for j in range(len(table_finite_diff[i])-1):
        dyi = float(f"{(table_finite_diff[i][j+1] - table_finite_diff[i][j]):.4f}")
        diy.append(dyi)
    table_finite_diff.append(diy)
    
df_1 = pd.DataFrame(data=table_finite_diff, index=['x_i', 'y_i', 'Δy_i', 'Δ2y_i', 'Δ3y_i', 'Δ4y_i', 'Δ5y_i', 'Δ6y_i',
'Δ7y_i', 'Δ8y_i', 'Δ9y_i', 'Δ10y_i','Δ11y_i'])
df_1 = df_1.transpose().fillna(0)

if input("Show all the suporting points, function values and finite differences? (y/n)\n") == "y":
    print(df_1)

h = xi[1] - xi[0]
mean = (xi[0] + xi[-1]) / 2

x1_data = interpPhase(x1_custom, h, mean)
x1_data = newtonFormula(x1_data)

x2_data = interpPhase(x2_custom, h, mean)
x2_data = newtonFormula(x2_data)

labels_row = ['x_1', 'x_2']
labels_col = ['Newton_Formula', 'SupportingPoint_Index', 'Phase_Of_Interp', 'Func_Derivative-1', 'Func_Derivative-2']
table_x = [x1_data, x2_data]

df_2 = pd.DataFrame(table_x, labels_row, labels_col)
print("\nFinal table with the results: ")
print(df_2)

if input("\nShow the graphs? (y/n)\n") == "y":
    plt.figure()
    plt.grid()
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Graph of the function')
    #plt.axhline(c='black')
    #plt.axvline(c='black')
    f = interp1d(xi, yi)
    args = [x1_custom, x2_custom]
    plt.plot(xi, f(xi), c='black', linewidth=1.5)
    plt.scatter(xi, yi, marker='o', c='red')
    plt.scatter(args, f(args), marker='o', c='b')
    plt.show()
