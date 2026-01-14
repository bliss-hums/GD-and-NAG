'''This script runs an error analysis for GD and NAG with and without backtracking, 
   as well as NAG with Adaptive Restart on the log-sum-exp function.'''

import numpy as np
import matplotlib.pyplot as plt


def implementation(rho, f, grad_f):
    '''This function runs the algorithms for all the methods (GD, NAG, GD w/ backtracking, NAG w/ backtracking and NAG w/ Adaptive Restart). 
       It then computes and returns the error analysis to be graphed.'''
    def gradient_descent(start, ts, iterations, eps):
        '''This function runs Gradient Descent.'''
        x_0 = start
        x_s = [start]
        k = 0
        norm_grad = np.dot(grad_f(x_0), grad_f(x_0)) # ||grad_f1||^2 to be used as stopping criteria.
        while k<iterations and norm_grad>eps:
            grad = grad_f(x_0)
            x_next = x_0 - (ts*grad)
            x_s.append(x_next)
            x_0 = x_next
            norm_grad = np.dot(grad, grad)
            k+=1
        return x_next, x_s #returns the list containing all x_k as well as the last x_k generated.
        
    def nag(start, ts, iterations, eps):
        '''This functions runs Nesterov's Accelerated Gradient.'''
        x_0 = y1 = x_0_gd = start
        x_s = y_s = x_s_gd = [start]
        t_1 = 1
        k = 0
        norm_grad = np.dot(grad_f(x_0), grad_f(x_0)) # ||grad_f1||^2 to be used as stopping criteria.
        while k<iterations and norm_grad>eps:
            grad = grad_f(y1)
            x_next = y1 - (ts*grad)
            x_s.append(x_next)
            t_next = 0.5 + np.sqrt((1/4)+ (t_1)**2)
            alpha = (t_1 - 1)/t_next
            y_next = x_next + (alpha*(x_next - x_0))
            x_0, t_1, y1 = x_next, t_next, y_next

            
            norm_grad = np.dot(grad_f(x_0), grad_f(x_0))
            k+=1
        return x_next, x_s #returns the list containing all x_k as well as the last x_k generated.

    def gd_bt(start, ts, iterations, eps):
        '''This function runs GD with backtracking.See Section 4.3 of the technical report.'''
        x_0 = start
        x_s = [start]
        k = 0
        theta = 0.8
        norm_grad = np.dot(grad_f(x_0), grad_f(x_0)) # ||grad_f1||^2 to be used as stopping criteria.
        while k<iterations and norm_grad>eps:
            grad = grad_f(x_0)
            i = 0
            while True: #finding smallest i_k to set ts_k s.that f(x_{k+1})<=Q_L(x_{k+1},x_k).
                ts_next = (theta**i)*ts
                x_next = x_0 - (ts_next*grad)
                dot = np.dot((x_next - x_0), grad)
                h = (1/(2*ts_next))*np.dot(x_next-x_0, x_next-x_0)
                if f(x_next) <= f(x_0) + dot + h:#f(x_0) + dot + h is Q_L(x_{k+1},x_k)
                    break
                i+=1
            x_s.append(x_next)
            ts = ts_next
            x_0 = x_next
            norm_grad = np.dot(grad, grad)
            k+=1
        return x_next, x_s

    def nag_bt(start, ts, iterations, eps):
        '''This function runs NAG with backtracking. See Section 4.3 of the technical report.'''
        x_0 = y1 = start
        x_s = [start]
        ts_hist = [ts]
        t_1 = 1
        k = 0
        theta = 0.8
        norm_grad = np.dot(grad_f(x_0), grad_f(x_0)) # ||grad_f1||^2 to be used as stopping criteria.
        while k<iterations and norm_grad>eps:
            grad = grad_f(y1)
            i = 0
            while True: #finding smallest i_k to set ts_k s.that f(x_{k+1})<=Q_L(x_{k+1},y_k).
                ts_next = (theta**i)*ts
                x_next = y1 - (ts_next*grad)
                dot = np.dot((x_next - y1), grad)
                h = (1/(2*ts_next))*np.dot(x_next-y1, x_next-y1)
                if f(x_next) <= f(y1) + dot + h: #f(y1) + dot + h is Q_L(x_{k+1},y_k).
                    break
                i += 1
            x_s.append(x_next)
            ts = ts_next
            ts_hist.append(ts_next)
            ts = ts_next
            t_next = 0.5 + np.sqrt((1/4)+ (t_1)**2)
            alpha = (t_1 - 1)/t_next
            y_next = x_next + (alpha*(x_next - x_0))
            x_0, t_1, y1 = x_next, t_next, y_next
   
            norm_grad = np.dot(grad_f(x_next), grad_f(x_next))
            k+=1
        return x_next, x_s

    def adaptive_restart(start, ts, iterations, eps):
        '''This function runs the adaptive restart on NAG. See Algorithm 5 and Section 4.4 in the technical report.
        This is the restart described in the O'Donoghue&Candes(2013) paper and not the restart we proposed which is run on the quadratic function.'''
        x_0 = y1 = start
        x_s = [start]
        t_1 = 1
        k = 0
        O_mthd = []
        norm_grad = np.dot(grad_f(x_0), grad_f(x_0)) # ||grad_f1||^2 to be used as stopping criteria.
        while k<iterations and norm_grad>eps:
            grad = grad_f(y1)
            x_next = y1 - (ts*grad)
            if f(x_next)>f(x_0):
                y_next = x_next
                t_next = 1
            else:
                t_next = 0.5 + np.sqrt((1/4)+ (t_1)**2)
                alpha = (t_1 - 1)/t_next
                y_next = x_next + (alpha*(x_next - x_0))
            x_s.append(x_next)
            x_0, t_1, y1 = x_next, t_next, y_next

            
            norm_grad = np.dot(grad, grad)
            k+=1
        return x_next, x_s
    
    def outputs(x_list, x_last):
        '''This function calculates the error of the various methods on the objective function. 
        Error = (f(x_k) - f(x_K))/f(x_K), where K is the last iteration for which the algorithm was ran.'''
        f_last = f(x_last)
        err = []
        for i in range(len(x_list)):
            f_k = f(x_list[i])
            err_k = (f_k - f_last)/f_last
            
            err.append(err_k)
            
        err = err[:-1]
        return err
    
    #all the algorithms are run and the error is then calculated for each method.
    mini_gd, liste_gd= gradient_descent(x, ts, 5000, 1e-16)
    err_gd = outputs(liste_gd, mini_gd)
    
    mini, liste = nag(x, ts, 5000, 1e-16)
    err = outputs(liste, mini)

    mini_gd_bt, liste_gd_bt = gd_bt(x, ts, 5000, 1e-16)
    err_gd_bt = outputs(liste_gd_bt, mini_gd_bt)

    mini_nag_bt, liste_nag_bt = nag_bt(x, ts, 5000, 1e-16)
    err_nag_bt = outputs(liste_nag_bt, mini_nag_bt)

    mini_rt, liste_rt = adaptive_restart(x,ts, 5000, 1e-16)
    err_rt = outputs(liste_rt, mini_rt)
    return err, err_gd, err_gd_bt, err_nag_bt, err_rt #error values for each method, to be graphed.

#initialization of variables
m = 100
n = 20
A = np.random.randn(m,n)
b = np.random.randn(m)
ts = 1/np.max(np.linalg.eigvals(A.T@A))
x = np.random.randn(n) #x_0
rho_list = [0.05,0.1,0.5,1]

#GRAPHING
fig, axes = plt.subplots(2, 2, figsize=(18,10), sharex=False)
axes = axes.flatten()
for i, rho in enumerate(rho_list):
    f = lambda x: rho*np.log(np.sum(np.exp((A@x-b)/rho)))
    grad_f = lambda x: A.T@(np.exp((A@x - b)/rho)/np.sum(np.exp((A@x-b)/rho))) 
    err, err_gd, err_gd_bt, err_nag_bt, err_rt= implementation(rho, f, grad_f)
    axes[i].plot(err_gd, 'k', label= 'GD')
    axes[i].plot(err, 'b', label= 'NAG')
    axes[i].plot(err_gd_bt, 'c--', label='GD with backtracking')
    axes[i].plot(err_nag_bt, 'y--', label='NAG with backtracking')
    axes[i].plot(err_rt, 'r', label= 'Adaptive restart')
    axes[i].set_ylabel('f_k - f*')   
    axes[i].set_xlabel('k')
    axes[i].set_yscale('log')
    axes[i].set_title(f'rho = {rho}', fontsize=10)
    axes[i].legend()


plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
    