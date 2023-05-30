#metod_easy_iteration
import numpy as np
from math import *

# наши границы a,b,c,d= 0,2,0,1 

#задаем u и f
def u(x: float,y: float)-> float:
    return sin(pi*x*y)

def f_main(x: float,y: float)-> float:
    return -exp(-x*y**2)

def f_test(x: float,y: float)-> float:
    return sin(pi*x*y)*((pi**2)*x**2+(pi**2)*y**2)
    

# задаем граничные условия
def mu1_main(y: float)-> float:
    return (y-2)*(y-3)
def mu2_main(y: float)-> float:
    return y*(y-2)*(y-3)
def mu3_main(x: float)-> float: 
    return (x-1)*(x-2)
def mu4_main(x: float)-> float: 
    return x*(x-1)*(x-2)

def mu1_test(y: float, a=1)-> float:
    return u(a,y)
def mu2_test(y: float, b=2)-> float:
    return u(b,y)
def mu3_test(x: float, c=2)-> float:
    return u(x,c)
def mu4_test(x: float, d=3)-> float:
    return u(x,d)

#метод чебышева
def Chebishev_metod(h2,k2,n,m,k_num,A,v,f,eps,N_max,counter_iterations, counter_steps,x,y):
    file = open('D:\\3_kurs\\test-2.txt','w')
    eps_max = 0  # maximum of eps
    flag = False  

    norm_r = 0  # norm of residual
    norm_z = 0  # norm of error
    #find eigen_value and tau
    eigen_value_min = -4 * (h2 * np.sin(np.pi / (2.0 * n)) ** 2 + k2 * np.sin(np.pi / (2.0 * m)) ** 2)
    eigen_value_max = -4 * (h2 * np.cos(np.pi / (2.0 * n)) ** 2 + k2 * np.cos(np.pi / (2.0 * m)) ** 2)
    
    tau = np.array([
        2 / (eigen_value_max + eigen_value_min + (eigen_value_max - eigen_value_min) * np.cos(
            np.pi * (2 * i + 1) / (2.0 * k_num))) for i in range(k_num)
    ])  
    r = np.zeros((n + 1, m + 1))  # residual

    while not flag:
        for j in range(1, m):
            for i in range(1, n):
                r[i, j] = A * v[i][j] + h2 * (v[i + 1][j] + v[i - 1][j]) + k2 * (v[i][j + 1] + v[i][j - 1]) + f[i][j]
        eps_max = 0
        for j in range(1, m):
            for i in range(1, n):
                v_old = v[i, j]
                v_new = v_old - tau[counter_steps] * r[i, j]
                eps_cur = abs(v_new - v_old)
                if eps_cur > eps_max:
                    eps_max = eps_cur
                v[i, j] = v_new
        counter_steps += 1
        if counter_steps == k_num:
            counter_steps = 0
            counter_iterations += 1

        if eps_max < eps or counter_iterations >= N_max:
            if counter_steps == 0:
                flag = True


    for j in range(1, m):
        for i in range(1, n):
            norm_r += pow(A * v[i][j] + h2 * (v[i + 1][j] + v[i - 1][j]) + k2 * (v[i][j + 1] + v[i][j - 1]) + f[i][j],
                          2)
            norm_z += pow(v[i][j] - u(x[i], y[j]), 2)

    norm_r = np.sqrt(norm_r)
    norm_z = np.sqrt(norm_z)
    #exact = np.array([np.array([u(x[i], y[j]) for j in range(m + 1)]) for i in range(n + 1)])  # u(x, y)
    file.write('Metod Chebisheva\n')
    for i in range(n+1):
        file.write('V')
        file.write(str(i)+'   ')
        for j in range(m+1):
            file.write(str(round(v[i][j],14)))
            if len(str(round(v[i][j],14))):
                help=18-len(str(round(v[i][j],14)))
                file.write(' '*help)
            file.write(' ')
        file.write('\n')
    return(v)

#Метод простых итераций
def Simple_metod(h2,k2,n,m,k_num,A,v,f,eps,N_max,counter_iterations, x,y):
    file = open('D:\\3_kurs\\test-2.txt','a')
    eps_max = 0  # maximum of eps
    flag = False  

    norm_r = 0  # norm of residual
    norm_z = 0  # norm of error
    #find eigen_value and tau
    eigen_value_min = -4 * (h2 * np.sin(np.pi / (2.0 * n)) ** 2 + k2 * np.sin(np.pi / (2.0 * m)) ** 2)
    eigen_value_max = -4 * (h2 * np.cos(np.pi / (2.0 * n)) ** 2 + k2 * np.cos(np.pi / (2.0 * m)) ** 2)
    
    tau = 2 / (eigen_value_max + eigen_value_min) 
    r = np.zeros((n + 1, m + 1))  # residual

    while not flag:
        for j in range(1, m):
            for i in range(1, n):
                r[i, j] = A * v[i][j] + h2 * (v[i + 1][j] + v[i - 1][j]) + k2 * (v[i][j + 1] + v[i][j - 1]) + f[i][j]
        eps_max = 0
        for j in range(1, m):
            for i in range(1, n):
                v_old = v[i, j]
                v_new = v_old - tau * r[i, j]
                eps_cur = abs(v_new - v_old)
                if eps_cur > eps_max:
                    eps_max = eps_cur
                v[i, j] = v_new
        counter_iterations+=1

        if eps_max < eps or counter_iterations >= N_max:
            flag = True
    for j in range(1, m):
        for i in range(1, n):
            norm_r += pow(A * v[i][j] + h2 * (v[i + 1][j] + v[i - 1][j]) + k2 * (v[i][j + 1] + v[i][j - 1]) + f[i][j],
                          2)
            norm_z += pow(v[i][j] - u(x[i], y[j]), 2)

    norm_r = np.sqrt(norm_r)
    norm_z = np.sqrt(norm_z)
   # exact = np.array([np.array([u(x[i], y[j]) for j in range(m + 1)]) for i in range(n + 1)])  # u(x, y)
    file.write('Metod Simple iteration\n')
    for i in range(n+1):
        file.write('V')
        file.write(str(i)+'   ')
        for j in range(m+1):
            file.write(str(round(v[i][j],14)))
            if len(str(round(v[i][j],14))):
                help=18-len(str(round(v[i][j],14)))
                file.write(' '*help)
            file.write(' ')
        file.write('\n')
    return(v)


def start_all_metod(n, m, N_max, eps, a, b, c, d,  k_num=4):
    
    file = open('D:\\3_kurs\\test-1.txt','w')
    h = (b - a) / (n)  
    k = (d - c) / (m)  
    x = np.linspace(a, b, n + 1)  
    y = np.linspace(c, d, m + 1)  
    f = np.array([np.array([-f_main(x[i], y[j]) for j in range(m + 1)]) for i in range(n + 1)])  # f(x, y)
    mu_1 = np.array([mu1_main(i) for i in y])  
    mu_2 = np.array([mu2_main(i) for i in y])  
    mu_3 = np.array([mu3_main(i) for i in x])  
    mu_4 = np.array([mu4_main(i) for i in x])  
    
    v = np.zeros((n + 1, m + 1))  
    
    counter_steps = 0  
    counter_iterations = 0  
    h2 = 1 / h ** 2  
    k2 = 1 / k ** 2  
    A = -2 * (h2 + k2)  
    
    v[0, :] = mu_1  
    v[n, :] = mu_2  
    v[:, 0] = mu_3  
    v[:, m] = mu_4  
    print(v)
    # linear interpolation by x
    for j in range(1, m):
        v[1:n, j] = mu_1[j] + (mu_2[j] - mu_1[j]) / (b - a) * (x[1:n] - a)
    print(v)
    
    #start Метод Чебышева
    v_Cheb=Chebishev_metod(h2,k2,n,m,k_num,A,v,f,eps,N_max,counter_iterations, counter_steps,x,y)
    print('V Чебышева =',v_Cheb)

    #start Метод простых итераций
    v_simple=Simple_metod(h2,k2,n,m,k_num,A,v,f,eps,N_max, counter_iterations,x,y)
    print('V простых итераций =', v_simple)


start_all_metod(10,10,1000,10**(-6),1,2,2,3)










