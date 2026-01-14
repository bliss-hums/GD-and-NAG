'''This script runs an error analysis for Proximal Gradient and NAG on the LASSO function.
See Example 3 in the technical report. '''

import numpy as np
import matplotlib.pyplot as plt

def sparse_vector(n, s, seed=None):
    '''Creates a vector with n entries, s of which are non-zero.'''
    rng = np.random.default_rng(seed)  # reproducible random numbers if seed given
    x = np.zeros(n)                    # start with all zeros
    idx = rng.choice(n, s, replace=False)  # pick s unique positions
    x[idx] = rng.normal(s)              # assign random values at those positions
    return x

def tao_operator(alpha, x):
    '''This is how x_{k+1} is calculated in Prox Gradient'''
    return np.sign(x)*np.maximum(np.abs(x)-alpha, 0)

def proximal_gradient(start, iterations, eps):
    '''This function runs the Proximal Gradient algorithm instead of Gradient Descent
    since the function has a non-smooth part.'''
    x_0 = start
    x_s = [start]
    k = 0
    norm_grad = np.dot(grad_f(x_0), grad_f(x_0))
    while k<iterations and norm_grad>eps:
        h = x_0 - (t*grad_f(x_0))
        x_next = tao_operator(rho*t, h)
        x_s.append(x_next)
        x_0 = x_next
        norm_grad = np.dot(grad_f(x_0), grad_f(x_0))
        k+=1
    return x_next, x_s

def proximal_nag(start, iterations, eps):
    x_0 = y1 = start
    x_s = [start]
    t_1 = 1
    k = 0
    norm_grad = np.dot(grad_f(x_0), grad_f(x_0))
    while k<iterations and norm_grad>eps:
        h = y1 - (t*grad_f(y1))
        x_next = tao_operator(rho*t, h)
        x_s.append(x_next)
        t_next = 0.5 + np.sqrt((1/4)+ (t_1)**2)
        alpha = (t_1 - 1)/t_next
        y_next = x_next + (alpha*(x_next - x_0))
        x_0, t_1, y1 = x_next, t_next, y_next
        norm_grad = np.dot(grad_f(x_0), grad_f(x_0))
        k+=1
    return x_next, x_s

def adaptive_restart(start, iterations, eps):
    x_0 = y1 = start
    x_s = [start]
    t_1 = 1
    k = 0
    norm_grad = np.dot(grad_f(x_0), grad_f(x_0))
    while k<iterations and norm_grad>eps:
        h = y1 - (t*grad_f(y1))
        x_next = tao_operator(rho*t, h)
        if f(x_next)>f(x_0):
            y_next = x_next
        else:
            t_next = 0.5 + np.sqrt((1/4)+ (t_1)**2)
            alpha = (t_1 - 1)/t_next
            y_next = x_next + (alpha*(x_next - x_0))
        x_s.append(x_next)
        x_0, t_1, y1 = x_next, t_next, y_next
        norm_grad = np.dot(grad_f(x_0), grad_f(x_0))
        k+=1
    return x_next, x_s

def outputs(x_last, x_list):
        f_last = f(x_last)
        err = []
        for i in range(len(x_list)):
            f_k = f(x_list[i])
            err_k = (f_k - f_last)/f_last
            
            err.append(err_k)
            
        err = err[:-1]
        return err


#construction of A and b with settings from the O'Donoghue&Candes(2013) paper on Adaptive Restart.
m1, n1, s1, rho = 500, 2000, 100, 1
A = np.random.randn(m1,n1)
w = np.random.normal(loc=0.0, scale=np.sqrt(0.1), size=m1)
y = sparse_vector(n1, s1)
b = A@y + w

t = 1 / (np.max(np.linalg.eigvalsh(A.T@A)))
x = np.random.randn(n1)
f = lambda y:(0.5*np.dot((A@y - b), (A@y - b))) + rho*np.sum(np.abs(y))
grad_f = lambda y:A.T@(A@y - b)

x_last, x_list= proximal_gradient(x, 10000, 1e-16)
x_last_nag, x_list_nag = proximal_nag(x, 10000, 1e-16)
x_last_rt, x_list_rt = adaptive_restart(x, 5000, 1e-16)
err = outputs(x_last, x_list)
err_nag = outputs(x_last_nag, x_list_nag)
err_rt = outputs(x_last_rt, x_list_rt)

#GRAPHING
fig, ax = plt.subplots(figsize=(6,6))
ax.plot(err, 'k', label = 'Proximal Gradient')
ax.plot(err_nag, 'b', label = 'NAG')
ax.plot(err_rt, '--r', label = 'Adaptive Restart')
ax.set_yscale('log')
ax.set_ylabel('f_k - f*')   
ax.set_xlabel('k')
ax.legend()

plt.show()