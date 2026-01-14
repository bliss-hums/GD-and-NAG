'''This script runs an error analysis for GD and NAG using a fixed tau = 1/L 
and then using an optimal stepsize on a strongly-convex, quadratic function
as outlined in Section 4.2 of the technical report.'''

import numpy as np
import matplotlib.pyplot as plt

def implementation(M, vec, f, grad_f):
    '''This function runs the algorithms for all the methods. 
       It then computes and returns the error analysis to be graphed.'''

    def step_size(M, grad):
        num = (grad.T@grad)
        denom = (grad.T@M@grad)
        if denom <= 1e-12:
            return 1/(np.max(np.linalg.eigvals(M)))
        return num/denom
        
    def gd_exact(start, iterations, eps):
        x_0 = start
        x_s = [start]
        k = 0
        norm_grad = np.dot(grad_f(x_0), grad_f(x_0)) # ||grad_f1||^2 to be used as stopping criteria.
        while k<iterations and norm_grad>eps:
            grad = grad_f(x_0)
            ts = step_size(M, grad)
            x_next = x_0 - (ts*grad)
            x_s.append(x_next)
            x_0 = x_next
            norm_grad = np.dot(grad, grad)
            k+=1
        return x_next, x_s

    def gradient_descent(start, ts, iterations, eps):
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
        return x_next, x_s

    def nag_exact(start, q, iterations, eps):
        x_0 = y1 = start
        x_s = y_s = [start]
        t_1 = 1
        k = 0
        norm_grad = np.dot(grad_f(x_0), grad_f(x_0)) # ||grad_f1||^2 to be used as stopping criteria.
        while k<iterations and norm_grad>eps:
            grad = grad_f(y1)
            ts = step_size(M, grad)
            x_next = y1 - (ts*grad)
            x_s.append(x_next)
            t_next = 0.5 * ((q-t_1**2) + np.sqrt((t_1**2 - q)**2 + 4 * t_1**2))
            alpha = (t_1 - t_1**2)/(t_1**2 + t_next)
            y_next = x_next + (alpha*(x_next - x_0))
            x_0, t_1, y1 = x_next, t_next, y_next

            
            norm_grad = np.dot(grad_f(x_next), grad_f(x_next))
            k+=1
        return x_next, x_s

    def gradient_descent_nag(start, ts, q, iterations, eps):
        x_0 = y1 = start
        x_s = y_s = [start]
        t_1 = 1
        k = 0
        norm_grad = np.dot(grad_f(x_0), grad_f(x_0)) # ||grad_f1||^2 to be used as stopping criteria.
        while k<iterations and norm_grad>eps:
            grad = grad_f(y1)
            x_next = y1 - (ts*grad)
            x_s.append(x_next)
            t_next = 0.5 * ((q-t_1**2) + np.sqrt((t_1**2 - q)**2 + 4 * t_1**2))
            alpha = (t_1 - t_1**2)/(t_1**2 + t_next)
            y_next = x_next + (alpha*(x_next - x_0))
            x_0, t_1, y1 = x_next, t_next, y_next

            
            norm_grad = np.dot(grad_f(x_next), grad_f(x_next))
            k+=1
        return x_next, x_s


    def outputs(x_list, x_last):
        x_exact = -np.linalg.inv(M)@ vec
        f_exact = f(x_exact)
        err = []
        for i in range(len(x_list)):
            f_k = f(x_list[i])
            err_k = (f_k - f_exact)
            
            err.append(err_k)
        return err
    
    ts = 1/(np.max(np.linalg.eigvals(M)))
    mu = np.min(np.linalg.eigvals(M))
    mini_gdex, liste_gdex = gd_exact(x, 10000, 1e-10)
    mini_nagex, liste_nagex = nag_exact(x,0,10000, 1e-10)
    mini_gd, liste_gd = gradient_descent(x, ts, 10000, 1e-10)
    mini_nag, liste_nag = gradient_descent_nag(x, ts, 0, 10000, 1e-10)
    mini_q, liste_q = gradient_descent_nag(x, ts, ts*mu, 10000, 1e-10)
    err_nag = outputs(liste_nag, mini_nag)
    err_q = outputs(liste_q, mini_q)
    err_gdex = outputs(liste_gdex, mini_gdex)
    err_nagex = outputs(liste_nagex, mini_nagex)
    err_gd = outputs(liste_gd, mini_gd)
    return err_gdex, err_gd, err_nag, err_q, err_nagex

#initialisation of variables
n = 4
c = np.ones(n)
delta = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

x = np.arange(1, n+1)

#GRAPHING
fig, axes = plt.subplots(2, 3, figsize=(18,10), sharex=False)
axes = axes.flatten()
for i in range(len(delta)):
    Q = np.eye(n)
    Q[-1, -1] = delta[i]
    f1 = lambda x: (0.5*x.T)@Q@x + c.T@x
    grad_f1 = lambda x: Q@x + c
    err_gdex, err_gd, err_nag, err_q, err_nagex = implementation(Q, c, f1, grad_f1)
    axes[i].plot(err_gd, '--k', label = 'GD')
    axes[i].plot(err_nag, 'b', label = 'NAG')
    axes[i].plot(err_gdex, 'r', label = 'GD exact')
    axes[i].plot(err_nagex, 'g', label = 'NAG exact')
    axes[i].plot(err_q, 'c', label = 'q = mu/L')
    axes[i].set_ylabel('f_k - f*')   
    axes[i].set_xlabel('k')
    axes[i].set_yscale('log')
    axes[i].set_title(f'delta = {delta[i]}', fontsize=10)
    axes[i].legend()

plt.tight_layout(rect=[0,0,1,0.97])
plt.show()