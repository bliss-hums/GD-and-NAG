'''This script runs an error analysis for GD and NAG with and without backtracking, 
   as well as NAG with q = mu/L and NAG with Adaptive Restart on a quadratic function.'''

import numpy as np
import matplotlib.pyplot as plt

def implementation(M, vec, f, grad_f):
    '''This function runs the algorithms for all the methods. 
       It then computes and returns the error analysis to be graphed.'''
    
    def step_size(M, grad):
        '''This function computes the exact step size to be used in the functions 'gd_exact' and 'nag_exact'.'''
        num = (grad.T@grad)
        denom = 2*(grad.T@M@grad)
        if denom <= 1e-12:
            return 1/(2*np.max(np.linalg.eigvals(M)))
        return num/denom
    
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
        return x_s #returns the list containing all x_k as well as the last x_k generated.

    def nag(start, ts, q, iterations, eps):
        '''This function runs NAG as outlined in Algorithm 4 of the technical report, with q = mu/L.'''
        x_0 = y1 = start
        x_s = [start]
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
        return x_s
    
    def gd_exact(start, iterations, eps):
        '''Runs Gradient Descent using an exact stepsize as outlined in 
        Section 4.2 of the report.'''

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
        return x_s
    
    def nag_exact(start, q, iterations, eps):
        '''Runs NAG using an exact stepsize as outlined in 
        Section 4.2 of the report.'''

        x_0 = y1 = start
        x_s =[start]
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
        return x_s
    
    def gd_bt(start, ts, iterations, eps):
        x_0 = start
        x_s = [start]
        k = 0
        theta = 0.8
        norm_grad = np.dot(grad_f(x_0), grad_f(x_0)) # ||grad_f1||^2 to be used as stopping criteria.
        while k<iterations and norm_grad>eps:
            grad = grad_f(x_0)
            i = 0
            while True:
                ts_next = (theta**i)*ts
                x_next = x_0 - (ts_next*grad)
                dot = np.dot((x_next - x_0), grad)
                h = (1/(2*ts_next))*np.dot(x_next-x_0, x_next-x_0)
                if f(x_next) <= f(x_0) + dot + h:
                    break
                i+=1
            x_s.append(x_next)
            ts = ts_next
            x_0 = x_next
            norm_grad = np.dot(grad, grad)
            k+=1
        return x_s
    
    def nag_bt(start, ts, iterations, eps):
        x_0 = y1 = start
        x_s = [start]
        t_1 = 1
        k = 0
        theta = 0.8
        norm_grad = np.dot(grad_f(x_0), grad_f(x_0)) # ||grad_f1||^2 to be used as stopping criteria.
        while k<iterations and norm_grad>eps:
            grad = grad_f(y1)
            i = 0
            while True:
                ts_next = (theta**i)*ts
                x_next = y1 - (ts_next*grad)
                dot = np.dot((x_next - y1), grad)
                h = (1/(2*ts_next))*np.dot(x_next-y1, x_next-y1)
                if f(x_next) <= f(y1) + dot + h:
                    break
                i += 1
            x_s.append(x_next)
            ts = ts_next
            ts = ts_next
            t_next = 0.5 + np.sqrt((1/4)+ (t_1)**2)
            alpha = (t_1 - 1)/t_next
            y_next = x_next + (alpha*(x_next - x_0))
            x_0, t_1, y1 = x_next, t_next, y_next
   
            norm_grad = np.dot(grad_f(x_next), grad_f(x_next))
            k+=1
        return x_s
    
    def adaptive_restart(start, ts, iterations, eps):
        '''This function runs the adaptive restart algorithm. See algorithm 5 in the technical report.'''
        x_0 = y1 = start
        x_s = [start]
        alphas = []
        t_1 = 1
        k = 0
        norm_grad = np.dot(grad_f(x_0), grad_f(x_0)) # ||grad_f1||^2 to be used as stopping criteria.
        while k<iterations and norm_grad>eps:
            grad = grad_f(y1)
            x_next = y1 - (ts*grad)
            if f(x_next)>f(x_0): #set y_{k+1} = x_{k+1}
                y_next = x_next 
                t_next = 1
            else:
                t_next = 0.5 + np.sqrt((1/4)+ (t_1)**2)
                alpha = (t_1 - 1)/t_next
                alphas.append(alpha)
                y_next = x_next + (alpha*(x_next - x_0))
            x_s.append(x_next)
            x_0, t_1, y1 = x_next, t_next, y_next

            
            norm_grad = np.dot(grad_f(x_next), grad_f(x_next))
            k+=1
        return x_s, alphas

    def adaptive_restart_proposed(start, ts, iterations, eps):
        '''As seen in Section 4.4 of the report, we propose an alternative restart and this function runs it.
        The error behaves similarly as in the 'normal' restart.'''
        x_0 = y1 = start
        x_s = [start]
        alphas= []
        t_1 = 1
        k = 0
        norm_grad = np.dot(grad_f(x_0), grad_f(x_0)) # ||grad_f1||^2 to be used as stopping criteria.
        while k<iterations and norm_grad>eps:
            grad = grad_f(y1)
            x_next = y1 - (ts*grad)
            if f(x_next)>f(x_0): #set y_{k+1} = x_k
                x_next = x_0
                y_next = x_0
            else:
                t_next = 0.5 + np.sqrt((1/4)+ (t_1)**2)
                alpha = (t_1 - 1)/t_next
                alphas.append(alpha)
                y_next = x_next + (alpha*(x_next - x_0))
                x_s.append(x_next)
            x_0, t_1, y1 = x_next, t_next, y_next
            
            norm_grad = np.dot(grad_f(x_next), grad_f(x_next))
            k+=1
        return x_s, alphas
    
    def adaptive_restart_proposed_fixed(start, ts, alpha_thresh, iterations, eps):
        '''In this function, the alternative restart is run again but 
        with an alpha that is frozen once the restart criterion is fixed.'''
        x_0 = y1 = start
        x_s = [start]
        alphas= []
        t_1 = 1
        k = 0
        norm_grad = np.dot(grad_f(x_0), grad_f(x_0)) # ||grad_f1||^2 to be used as stopping criteria.
        while k<iterations and norm_grad>eps:
            grad = grad_f(y1)
            x_next = y1 - (ts*grad)
            if f(x_next)>f(x_0):
                x_next = x_0
                y_next = x_0
            else:
                t_next = 0.5 + np.sqrt((1/4)+ (t_1)**2)
                
                alpha = (t_1 - 1)/t_next
                if alpha>=alpha_thresh: #setting 
                    alpha = alphas[-1]
                alphas.append(alpha)
                y_next = x_next + (alpha*(x_next - x_0))
                x_s.append(x_next)
            x_0, t_1, y1 = x_next, t_next, y_next
            
            norm_grad = np.dot(grad_f(x_next), grad_f(x_next))
            k+=1
        return x_s, alphas
    
    
    def outputs(x_list):
        '''This function calculates the error of the various methods on the objective function. 
        Error = (f(x_k) - f(x*)), where x* is the exact minimizer of the function. 
        Here, we do not divide by f(x*) which is equal to 0.'''
        x_exact = -0.5* np.linalg.inv(M)@ vec
        f_exact = f(x_exact)
        err = []
        for i in range(len(x_list)):
            f_k = f(x_list[i])
            err_k = (f_k - f_exact)
            err.append(err_k)
            
        return err
 
    
    liste = nag(x,ts, 0, 5000, 1e-16)
    liste_q = nag(x,ts, q, 800, 1e-16)
    liste_gd = gradient_descent(x, ts, 5000, 1e-16)
    liste_bt = nag_bt(x,ts, 5000, 1e-16)
    liste_gd_bt = gd_bt(x, ts, 5000, 1e-16)
    liste_gd_exact = gd_exact(x, 5000, 1e-16)
    liste_exact = nag_exact(x, 0, 5000, 1e-16)
    liste_rt, alphas_rt = adaptive_restart(x, ts, 600, 1e-16)
    liste_rtp, alphas_rtp = adaptive_restart_proposed(x, ts, 1000, 1e-16)
    for i in range(1, len(alphas_rt)): #looking for first time restart criterion is reached, to freeze alpha at this point.
        if alphas_rt[i] == 0:
            i_dip = i-1
            break
    liste_rtpf, alphas_rtpf = adaptive_restart_proposed_fixed(x, ts, alphas_rt[i_dip], 1000, 1e-16)
    err = outputs(liste)
    err_q = outputs(liste_q)
    err_gd_exact = outputs(liste_gd_exact)
    err_exact = outputs(liste_exact)
    err_rtp = outputs(liste_rtp)
    err_rtpf = outputs(liste_rtpf)
    err_rt = outputs(liste_rt)
    err_gd = outputs(liste_gd)
    err_gd_bt = outputs(liste_gd_bt)
    err_bt = outputs(liste_bt)

    
    return err, err_q, err_bt, err_gd_exact, err_exact, err_gd, err_gd_bt, err_rt, err_rtp, err_rtpf, alphas_rtp, alphas_rtpf, alphas_rt

#initialization of variables
n = 20
_ = np.random.randn(n,n)
eigs = np.geomspace(1.0, 1000, n)
Q, _ = np.linalg.qr(np.random.randn(n, n))
A = Q @ np.diag(eigs) @ Q.T #this construction of A enables control of the eigenvalues, ensuring control of the condition number.

b = np.random.randn(n)
ts = 1/(2*np.max(np.linalg.eigvals(A)))
mu = np.min(np.linalg.eigvals(A))
q = mu*ts #L = 1/ts.
x = np.random.randn(n) #x_0

#GRAPHING
fig, ax = plt.subplots(figsize=(6,6))
c = 1
f = lambda x: x.T@A@x + b.T@x + c
grad_f = lambda x: (2*A)@x + b
err, err_q, err_bt, err_gd_exact, err_exact, err_gd, err_gd_bt, err_rt, err_rtp, err_rtpf, alphas_rtp, alphas_rtpf, alphas_rt = implementation(A, b, f, grad_f)
ax.plot(err_gd_bt, 'c', label = 'gd with backtracking')
ax.plot(err_bt, 'y', label = 'nag with backtracking')
ax.plot(err_gd, 'k--', label = 'gd')
ax.plot(err, 'b--', label = 'nag')
ax.plot(err_gd_exact, 'k', label = 'gd exact_t')
ax.plot(err_exact, 'b', label = 'nag exact_t')
ax.plot(err_rt, '--r', label = 'adaptive restart')
ax.plot(err_rtpf, 'g', label = 'proposed adaptive restart')
ax.plot(err_rtp, '#FFA500', label = 'proposed restart, fixed alpha')
ax.plot(err_q, 'm--', label = 'q = mu/L')
ax.set_ylabel('f_k - f*')   
ax.set_xlabel('k')
ax.set_yscale('log')
ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
