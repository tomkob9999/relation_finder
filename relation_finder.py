# Relation Finder

# Author: Tomio Kobayashi
# Version 1.0.5
# Updated: 2024/03/11

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Lasso

import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats as statss
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sp
from scipy.optimize import curve_fit

class relation_finder:
    
    
    def find_jumps(data):
        gaps = [np.abs(data[i] - data[i-1]) for i in range(1, len(data), 1)]
        sorted_data = sorted(gaps)
        num_items = len(sorted_data)
        exclude_each_end = int(num_items * 0.1)
        middle_80 = sorted_data[exclude_each_end:num_items-exclude_each_end]
        avg = np.mean(middle_80)
        return [i for i in range(1, len(data), 1) if np.abs(data[i] - data[i-1]) > avg*3]

    def remove_outliers(x, y):
        # Convert lists to numpy arrays if necessary
        x = np.array(x)
        y = np.array(y)

        # Function to calculate IQR and filter based on it
        def iqr_filter(data):
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (data >= lower_bound) & (data <= upper_bound)
        y_mask = iqr_filter(y)
        outs = [i for i, o in enumerate(y_mask) if o == False]
        if len(outs) > 0:
            print("Outliers skipped (lines):", outs)
        x_filtered = x[y_mask]
        y_filtered = y[y_mask]

        return x_filtered, y_filtered

    def exp_func(x, a, b, c):
        return (a+c*x) * np.exp(b * x)
    
    def poly_func(x, a, b, c):
        return a + c*x + b*x**2

    def fit_exp(x_data, y_data, func=None, init_guess=[]):
        if func is None:
            func = relation_finder.exp_func
        try:
            return curve_fit(func, x_data, y_data, method="dogbox", nan_policy="omit")
        except RuntimeError as e:
            try:
                guess = init_guess
                if len(guess) == 0:
                    guess = [min(y_data), 0, (max(y_data)-min(y_data))/len(y_data)]
                return curve_fit(func, x_data, y_data, method="dogbox", nan_policy="omit", p0=guess)
            except RuntimeError as ee:
                return None, None

    def fit_poly(x_data, y_data, init_guess=[]):
        return curve_fit(relation_finder.poly_func, x_data, y_data, method="dogbox", nan_policy="omit", p0=[min(y_data), 0, (max(y_data)-min(y_data))/len(y_data)])
        

    def find_relations(data, colX, colY, cols=[], const_thresh=0.1, skip_inverse=True, use_lasso=False, skip_outliers=False):

        if use_lasso:
            dic_relation = {
#                 0: ("P", "Proportional Linearly (Y=a*X)"),
#                 1: ("IP", "Inversely Proportional Linearly (Y=a*(1/X))"),
#                 2: ("QP", "Proportional by Square (Y=a*(X^2))"),
#                 3: ("IQP", "Inversely Proportional by Square (Y=a*(1/X^2))"),
#                 4: ("SP", "Proportional by Square Root (Y=a*sqrt(X)"),
#                 5: ("ISP", "Inversely Proportional by Square Root (Y=a*(1/sqrt(X))"),
                0: ("P", "Proportional Linearly (Y=a*X)"),
                1: ("IP", "Inversely Proportional Linearly (Y=a*(1/X))"),
                2: ("QP", "Proportional by exponential of 0.02 (Y=x*e^0.02)"),
                3: ("IQP", "Proportional by exponential of -0.02 (Y=x*e^-0.02)"),
                4: ("SP", "Proportional by exponential of 0.04 (Y=x*e^0.04)"),
                5: ("ISP", "Proportional by exponential of -0.04 (Y=x*e^-0.04)"),
            }
            num_incs = 6

            if skip_inverse:
                dic_relation = {
#                     0: ("P", "Proportional Linearly (Y=a*X)"),
#                     1: ("QP", "Proportional by Square (Y=a*(X^2))"),
#                     2: ("SP", "Proportional by Square Root (Y=a*sqrt(X)"),
                    0: ("P", "Proportional Linearly (Y=a*X)"),
                    1: ("QP", "Proportional by exponential of 0.02 (Y=x*e^0.02)"),
                    2: ("SP", "Proportional by exponential of -0.04 (Y=x*e^-0.04)"),
                }
                num_incs = 3

            xcol_size = len(data[0])-1
            if not skip_inverse:
                for i in range(xcol_size):
                    for row in data:
                        row.insert(-1, 1/row[i] if row[i] != 0 else 0)
            for i in range(xcol_size):
                for row in data:
#                     row.insert(-1, row[i] ** 2)
                    row.insert(-1, row[i] * np.e**-0.02)
            if not skip_inverse:
                for i in range(xcol_size):
                    for row in data:
#                         row.insert(-1, 1/row[i] ** 2)
                        row.insert(-1, row[i] * np.e**0.02)
            for i in range(xcol_size):
                for row in data:
#                     row.insert(-1, np.sqrt(row[i]) if row[i] > 0 else np.sqrt(row[i]*-1)*-1)
                    row.insert(-1, row[i] * np.e**-0.04)
            if not skip_inverse:
                for i in range(xcol_size):
                    for row in data:
#                         row.insert(-1, 1/np.sqrt(row[i]) if row[i] > 0 else 1/np.sqrt(row[i]*-1)*-1)
                        row.insert(-1, row[i] * np.e**0.04)

            model = Lasso()
            X_train = [r[:-1]for r in data]
            Y_train = [r[-1]for r in data]
            X_train, Y_train = relation_finder.remove_outliers(X_train, Y_train)
#             print("X_train", X_train)
#             print("Y_train", Y_train)
            model.fit(X_train, Y_train)

            print(f"Relation to {colY}")
            print("  Intersect:", model.intercept_)
#             print("  Coeffeicients:", model.coef_)
            print("  Coeffeicients:")

#             print("len(model.coef_)", len(model.coef_))
#             print("len(cols)", len(cols))
#             print("len(data[0])", len(data[0]))
#             print("num_incs", num_incs)
            numcols = int(len(model.coef_)/num_incs)
            for i, c in enumerate(model.coef_):
                if np.abs(c) > 0.0000001:
#                     print("i%num_incs", i%num_incs)
#                     print("int(i/num_incs)", int(i/num_incs))
#                     print("    ", cols[int(i/num_incs)] if len(cols) > 0 else "    Col" + str(int(i/num_incs)), ":", dic_relation[i%num_incs][1], round(c, 10))
                    print("    ", cols[i%numcols] if len(cols) > 0 else "    Col" + str(int(i/numcols)), ":", dic_relation[int(i/numcols)][1], round(c, 10))
            predictions = model.predict(X_train)
            r2 = r2_score(Y_train, predictions)
            print("  R2:", round(r2, 5))

            for i in range(numcols):
                pdata = [[row[i], row[-1]] for row in data]
                print("pdata", pdata)
                df = pd.DataFrame(pdata, columns=[cols[i], colY])
                plt.title("Scatter Plot of " + cols[i] + " and " + colY)
                plt.scatter(data=df, x=cols[i], y=colY)
                plt.figure(figsize=(3, 2))
                plt.show()

            return model.coef_.tolist() + [model.intercept_]

        else:
            
        # Fit a polynomial of the specified degree to the data
            X_train = [r[0]for r in data]
            Y_train = [r[-1]for r in data]
            
            jumps = relation_finder.find_jumps(Y_train)
            for jum in jumps:
                print("JUMP DETECTED at", colX, "(original scale)=", X_train[jum], ",", colY, "=", Y_train[jum])
            

            X_train_filtered, Y_train_filtered = relation_finder.remove_outliers(X_train, Y_train)
#             X_train_filtered = X_train
#             Y_train_filtered = Y_train
#             params, covariance = relation_finder.fit_exp(X_train_filtered, Y_train_filtered)
        
            # reduced to less than 10 so that exponential can be used in regression
            div = 10**int(np.log10(max(X_train)))

            use_gaussian = False
            my_exp_func1 = lambda x, a, b, c: (a+c*x/div) * np.exp(b * x/div)
            params, covariance = relation_finder.fit_exp(X_train_filtered, Y_train_filtered, func=my_exp_func1)
            if params is None:
                r2_1 = -999
            else:
                a1, b1, c1 = params
                predictions = [my_exp_func1(x, a1, b1, c1) for x in X_train]
                r2_1 = r2_score(Y_train, predictions)
            
            my_exp_func2 = lambda x, a, b, c: (a+c*x/div) * np.exp(b * (1 + x/div))
            params, covariance = relation_finder.fit_exp(X_train_filtered, Y_train_filtered, func=my_exp_func2)
            if params is None:
                r2_2 = -999
            else:
                a2, b2, c2 = params
                predictions = [my_exp_func2(x, a2, b2, c2) for x in X_train]
                r2_2 = r2_score(Y_train, predictions)
            
            poly_used = False
            
#             print("r2_1", r2_1)
#             print("r2_2", r2_2)
            
            if r2_1 == -999 and r2_2 == -999:
                params, covariance = relation_finder.fit_poly(X_train_filtered, Y_train_filtered)
                poly_used = True
                
                a, b, c = params
                predictions = [relation_finder.poly_func(x, a, b, c) for x in X_train]
                r2 = r2_score(Y_train, predictions)
            elif r2_1 > r2_2:
                a, b, c = a1, b1, c1
                r2 = r2_1
                my_exp_func = my_exp_func1
            else:
                use_gaussian = True
                a, b, c = a2, b2, c2
                r2 = r2_2
                my_exp_func = my_exp_func2
#             print("r2_1", r2_1)
#             print("r2_2", r2_2)
#             print("r2", r2)

            if skip_outliers:
                X_train_g = X_train_filtered
                Y_train_g = Y_train_filtered
            else:
                X_train_g = X_train
                Y_train_g = Y_train
                
            if np.abs(b) < const_thresh and np.abs(c) < const_thresh :
                print(f"{colY} is CONSTANT to {colX} with constant value of {a:.5f} with confidence level (R2) of {r2*100:.2f}%")
                return [colX, a, 0, 0]
            else:
                if c > 0 and not poly_used and b > 0.10:
                    print("   *   *   *   *   *")
                    print(f"EXPONENTIAL GROWH DETECTED {b:.5f}")
                    print("   *   *   *   *   *")
                if poly_used:
                    equation = f"y = {a:.8f} + {c:.8f}*x + {b:.8f}*x**2)"
                    print(f"Equation:", equation)
                else:
                    print(f"Intercept: {a:.5f}")
                    print(f"Slope (original scale): {c/div:.5f}")
                    print(f"Exponential Factor: {b:.5f}")
                    if use_gaussian:
                        equation = f"y = ({a:.8f} + {c:.8f}*(x/{div}))) * e**({b:.8f}*(1+x/{div}))"
                    else:
                        equation = f"y = ({a:.8f} + {c:.8f}*(x/{div}))) * e**({b:.8f}*(x/{div}))"
                    print(f"Equation:", equation)
                    print("R2:", round(r2, 5))
                pdata = [[x, Y_train_g[i]] for i, x in enumerate(X_train_g)]
                df = pd.DataFrame(pdata, columns=[colX, colY])
                plt.title("Scatter Plot of " + colX + " and " + colY)
                plt.scatter(X_train_g, Y_train_g)
                if not skip_outliers:
                    plt.scatter([x for i, x in enumerate(Y_train_g) if i in jumps or (i != 0 and i+1 in jumps)], [x for i, x in enumerate(Y_train_g) if i in jumps or (i != 0 and i+1 in jumps)], color='pink')
                plt.xlabel(colX)
                plt.ylabel(colY)
                # Generate x values for the line
                x_line = np.linspace(min(X_train_g), max(X_train_g), 1000)  # 100 points from min to max of scatter data
                y_line = [my_exp_func(x, a, b, c) if not poly_used else relation_finder.poly_func(x, a, b, c) for x in x_line]
                # Plot the line
                plt.plot(x_line, y_line, color='red', label='Line: ' + equation)
                plt.figure(figsize=(3, 2))
                plt.show()
                return [colX, a, c if poly_used else c/div, b]