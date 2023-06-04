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
def mu1_main(y: float,a)-> float:
    return (y-2)*(y-3)
def mu2_main(y: float,b)-> float:
    return y*(y-2)*(y-3)
def mu3_main(x: float,c)-> float: 
    return (x-1)*(x-2)
def mu4_main(x: float,d)-> float: 
    return x*(x-1)*(x-2)

def mu1_test(y: float, a)-> float:
    return u(a,y)
def mu2_test(y: float, b)-> float:
    return u(b,y)
def mu3_test(x: float, c)-> float:
    return u(x,c)
def mu4_test(x: float, d)-> float:
    return u(x,d)

#метод чебышева
def Chebishev_metod(h2,k2,n,m,k_num,A,v,f,eps,N_max,counter_iterations, counter_steps,x,y,F):
    
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
    exact = 0
   # file.write('Metod Chebisheva\n')
    #for i in range(n+1):
    #    file.write('V')
     #   file.write(str(i)+'   ')
      #  for j in range(m+1):
       #     file.write(str(round(v[i][j],14)))
        #    if len(str(round(v[i][j],14))):
         #       help=18-len(str(round(v[i][j],14)))
          #      file.write(' '*help)
           # file.write(' ')
        #file.write('\n')
    return(v,exact, counter_iterations, eps_max, norm_r, norm_z)

#Метод простых итераций
def Simple_metod(h2,k2,n,m,k_num,A,v,f,eps,N_max,counter_iterations, x,y,F):
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
            #norm_z += pow(v[i][j] - u(x[i], y[j]))

    norm_r = np.sqrt(norm_r)
    norm_z = np.sqrt(norm_z)
    exact = 0
    #file.write('Metod Simple iteration\n')
    #for i in range(n+1):
     #   file.write('V')
      #  file.write(str(i)+'   ')
       # for j in range(m+1):
        #    file.write(str(round(v[i][j],14)))
         #   if len(str(round(v[i][j],14))):
          #      help=18-len(str(round(v[i][j],14)))
           #     file.write(' '*help)
            #file.write(' ')
        #file.write('\n')
    
    return(v,exact,counter_iterations, eps_max, norm_r, norm_z)


def start_main(n, m, N_max, eps, a, b, c, d, Var, F,mu1,mu2,mu3,mu4, k_num):
    h = (b - a) / (n)  
    k = (d - c) / (m)  
    x = np.linspace(a, b, n + 1)  
    y = np.linspace(c, d, m + 1)  
    f = np.array([np.array([F(x[i], y[j]) for j in range(m + 1)]) for i in range(n + 1)])  # f(x, y)
    mu_1 = np.array([mu1(i,a) for i in y])  
    mu_2 = np.array([mu2(i,b) for i in y])  
    mu_3 = np.array([mu3(i,c) for i in x])  
    mu_4 = np.array([mu4(i,d) for i in x])  
    
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

    # linear interpolation by x
    for j in range(1, m):
        v[1:n, j] = mu_1[j] + (mu_2[j] - mu_1[j]) / (b - a) * (x[1:n] - a)

    #невязка на начальном приближении
    R0=0
    for i in range(len(f)):
        for j in range(len(f[i])):
            if R0<abs(v[i,j]-f[i,j]):
                R0=abs(v[i,j]-f[i,j])
    
    if Var==1:
        v_method,exact,counter_iterations, eps_max, norm_r, norm_z=Chebishev_metod(h2,k2,n,m,k_num,A,v,f,eps,N_max,counter_iterations, counter_steps,x,y,F)
        print(1)
    elif Var==2:
        v_method,exact,counter_iterations, eps_max, norm_r, norm_z=Simple_metod(h2,k2,n,m,k_num,A,v,f,eps,N_max, counter_iterations,x,y,F)
    ### теперь половинный шаг
    n_half=n*2
    m_half=m*2
    h_half = (b - a) / (n_half)  
    k_half = (d - c) / (m_half)  
    x_half = np.linspace(a, b, n_half + 1)  
    y_half = np.linspace(c, d, m_half + 1)  
    f_half = np.array([np.array([F(x_half[i], y_half[j]) for j in range(m_half + 1)]) for i in range(n_half + 1)])  # f(x, y)
    mu_1_half = np.array([mu1(i,a) for i in y_half])  
    mu_2_half = np.array([mu2(i,b) for i in y_half])  
    mu_3_half = np.array([mu3(i,c) for i in x_half])  
    mu_4_half = np.array([mu4(i,d) for i in x_half])  
    
    v_half = np.zeros((n_half + 1, m_half + 1))  
    
    counter_steps_half = 0  
    counter_iterations_half = 0  
    h2_half = 1 / h_half ** 2  
    k2_half = 1 / k_half ** 2  
    A_half = -2 * (h2_half + k2_half)  
    
    v_half[0, :] = mu_1_half 
    v_half[n_half, :] = mu_2_half  
    v_half[:, 0] = mu_3_half  
    v_half[:, m_half] = mu_4_half  
    #print(v)
    # linear interpolation by x
    for j in range(1, m_half):
        v_half[1:n_half, j] = mu_1_half[j] + (mu_2_half[j] - mu_1_half[j]) / (b - a) * (x_half[1:n_half] - a)
    #print(v)
    if Var==1:
        v_method_half,exact_half,counter_iterations_half, eps_max_half, norm_r_half, norm_z_half=Chebishev_metod(h2_half,
             k2_half,n_half,m_half,k_num,A_half,v_half,f_half,eps,N_max,counter_iterations_half, counter_steps_half,x_half,y_half,F)
    elif Var==2:
        v_method_half,exact_half,counter_iterations_half, eps_max_half, norm_r_half, norm_z_half=Simple_metod(h2_half,
        k2_half,n_half,m_half,k_num,A_half,v_half,f_half,eps,N_max, counter_iterations_half,x_half,y_half,F)
    v_half_out= np.zeros((int(n) + 1, int(m) + 1))
    for i in range(0, n*2 + 1, 2):
        for j in range(0, m*2 + 1, 2):
            v_half_out[int(i / 2), int(j / 2)] = v_method_half[i, j]
    ### половинный шаг окончен 
    v=[[0 for i in range (n+1)] for j in range(m+1)]
    for i in range(m+1):
        for j in range(n+1):
            v[j][i]=round(v_method[j][i],15)
    
    return v,v_half_out,abs(v-v_half_out),x,y,counter_iterations, counter_iterations_half, eps_max, norm_r, norm_z,R0
    
def start_test(n, m, N_max, eps, a, b, c, d, Var, F,mu1,mu2,mu3,mu4, k_num):
      
    h = (b - a) / (n)  
    k = (d - c) / (m)  
    x = np.linspace(a, b, n + 1)  
    y = np.linspace(c, d, m + 1)  
    f = np.array([np.array([F(x[i], y[j]) for j in range(m + 1)]) for i in range(n + 1)])  # f(x, y)
    mu_1 = np.array([mu1(i,a) for i in y])  
    mu_2 = np.array([mu2(i,b) for i in y])  
    mu_3 = np.array([mu3(i,c) for i in x])  
    mu_4 = np.array([mu4(i,d) for i in x])  
    
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
    
    

    #print(v)
    # linear interpolation by x
    for j in range(1, m):
        v[1:n, j] = mu_1[j] + (mu_2[j] - mu_1[j]) / (b - a) * (x[1:n] - a)
    
    #невязка на начальном приближении
    R0=0
    for i in range(len(f)): 
        for j in range(len(f[i])):
            if R0<abs(v[i,j]-u(i,j)):
                R0=abs(v[i,j]-u(i,j))
    print(R0)
    #print(v)
    if Var==1:
        v_method,exact,counter_iterations, eps_max, norm_r, norm_z=Chebishev_metod(h2,k2,n,m,k_num,A,v,f,eps,N_max,counter_iterations, counter_steps,x,y,F)
        print(1)
    elif Var==2:
        v_method,exact,counter_iterations, eps_max, norm_r, norm_z=Simple_metod(h2,k2,n,m,k_num,A,v,f,eps,N_max, counter_iterations,x,y,F)

    v=[[0 for i in range (n+1)] for j in range(m+1)]
    for i in range(m+1):
        for j in range(n+1):
            v[j][i]=round(v_method[j][i],12)
    
    exact = np.array([np.array([u(x[i], y[j]) for j in range(m + 1)]) for i in range(n + 1)])  # u(x, y)

    return v,exact,abs(v-exact),x,y,counter_iterations, eps_max, norm_r, norm_z,R0








