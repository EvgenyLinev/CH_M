#оформление
from mpl_toolkits.mplot3d import Axes3D
from sympy.solvers import solve
from sympy import Symbol
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math
from math import sin 
import numpy as np
from tkinter import *
from tkinter import scrolledtext
from tkinter import ttk
import sys
import metod_easy_iteration
from metod_easy_iteration import start_all_metod, mu1_main, mu2_main,mu3_main,mu4_main,\
                                f_main, f_test,mu1_test,mu3_test,mu4_test,mu2_test

def table(tab,f,mu1,mu2,mu3,mu4):
    #c = Canvas(tab, height=260, width = 500)
    #c.place(x='15',y='20')
    #c.create_rectangle(0, 0, 360, 500)
    if tab==tab1:
        u,x_3d,y_3d,counter_iterations, eps_max, norm_r, norm_z=start_all_metod(int(txt_n.get()),int(txt_m.get()),int(txt_N_max.get()),float(txt_eps.get()),
                        float(txt_a.get()),float(txt_b.get()),float(txt_c.get()),float(txt_d.get()), 
                        Var.get(),f,mu1,mu2,mu3,mu4, int(txt_k.get()))
    if tab==tab2:
        u,x_3d,y_3d,counter_iterations, eps_max, norm_r, norm_z=start_all_metod(int(txt_n_2.get()),int(txt_m_2.get()),int(txt_N_max_2.get()),float(txt_eps_2.get()),
                        float(txt_a_2.get()),float(txt_b_2.get()),float(txt_c_2.get()),float(txt_d_2.get()), 
                        Var.get(),f,mu1,mu2,mu3,mu4, int(txt_k_2.get()))
    table_u=[[i] for i in range (len(u))]
    for i in range (len(u)):
        for j in range(len(u[i])):
            table_u[i].append(u[i][j])

    
    heads=[]
    for i in range(len(table_u[i])):
        heads.append(str(i))
        
    table = ttk.Treeview(tab, show='headings', columns=heads, height=13)
    #table.pack(fill=X, padx=[25,15], pady=5)
    table.place(x='25', y='5', relwidth=0.95)

    table.heading("0", text="V (i\j)", anchor='w')
    table.column('#1', width=50)
    for i in range(1, len(table_u[0])):
        j=str(i-1)
        table.heading(str(i), text=j, anchor='w')
        table.column('#'+str(i), width=100)
    
    for person in table_u:
        table.insert("", END, values=person)

    scrollbary = Scrollbar(tab, orient=VERTICAL)
    scrollbarx = Scrollbar(tab, orient=HORIZONTAL)
    table.configure(yscrollcommand=scrollbary.set, xscrollcommand=scrollbarx.set)
    scrollbarx.configure(command=table.xview)
    scrollbary.configure(command=table.yview)
    scrollbary.place(x=5, y=30, width=15, height=250)  
    scrollbarx.place(x=650, y= 300, width=250, height=15)

    fig=plt.figure(figsize=(6,4))   
    #figure.add_subplot(111).plot(u[2], u[4], 'r-',u[2],u[3],'b-')
    ax=fig.add_subplot(projection='3d')
    xgrid,ygrid=np.meshgrid(x_3d,y_3d)
    u=np.matrix(u)
    ax.plot_surface(xgrid,ygrid,u)
    
    chart=FigureCanvasTkAgg(fig,tab)
    chart.get_tk_widget().place(x='800', y='320')
    ax.set_xlabel('x', color='gray')
    ax.set_ylabel('y',color='gray')
    ax.set_zlabel('z',color='gray')
    #plt.grid(True)

    win=Tk()
    win.title('Информация')
    win.geometry('500x500')
    win_tab=ttk.Notebook(win)
    win_txt=Label(win_tab, text=("Количество итераций: "+str(counter_iterations)))
    win_txt.place(x='5',y='15')
    win_txt=Label(win_tab, text="Итоговая точность: "+str(eps_max))
    win_txt.place(x='5',y='45')
    win_txt=Label(win_tab, text="Норма невязки: "+str(norm_r))
    win_txt.place(x='5',y='75')
    win_txt=Label(win_tab, text="Норма погрешности: "+str(norm_z))
    win_txt.place(x='5',y='105')
    win_tab.pack(expand=1, fill='both') 
    win.mainloop()

    

def exit():
    sys. exit()  

window = Tk()
window.title("Линев, Викулов, вторая ступень, Команда Викулова Максима")
window.geometry('1500x800')

tab_control = ttk.Notebook(window)  
tab1 = ttk.Frame(tab_control)  
tab2 = ttk.Frame(tab_control)  
tab_control.add(tab1, text='Основная задача')  
tab_control.add(tab2, text='Тестовая задача')  

btn_osn = Button(tab1, text="Нажмите на кнопку, чтобы построить решение", 
                 command=lambda: table(tab1,f_main, mu1_main, mu2_main,mu3_main,mu4_main,))
btn_osn.place(x='10', y='295')

btn_test = Button(tab2, text="Нажмите на кнопку, чтобы построить решение", 
                  command=lambda: table(tab2,f_test,mu1_test,mu2_test,mu3_test,mu4_test))
btn_test.place(x='10', y='295')

#далее работаем над окном для основной задачи
lbl = Label(tab1, text="Введите максимальное количество итераций N_max:", font=(18))
lbl.place(x='10', y='330')
txt_N_max = Entry(tab1, width=20)
txt_N_max.place(x='410', y='335')
txt_N_max.insert(0,'1000')

lbl = Label(tab1, text="Введите точность вычислений eps:", font=(18))
lbl.place(x='10', y='360')
txt_eps = Entry(tab1, width=20)
txt_eps.place(x='280', y='365')
txt_eps.insert(0,'0.000001')

lbl = Label(tab1, text="Границы заданы условием, но их можно поменять", font=(18))
lbl.place(x='10', y='390')

lbl = Label(tab1, text="a =", font=(18))
lbl.place(x='10', y='420')
txt_a = Entry(tab1, width=20)
txt_a.place(x='40', y='425')
txt_a.insert(0,'1')

lbl = Label(tab1, text="b =", font=(18))
lbl.place(x='10', y='450')
txt_b = Entry(tab1, width=20)
txt_b.place(x='40', y='455')
txt_b.insert(0,'2')

lbl = Label(tab1, text="c =", font=(18))
lbl.place(x='10', y='480')
txt_c = Entry(tab1, width=20)
txt_c.place(x='40', y='485')
txt_c.insert(0,'2')

lbl = Label(tab1, text="d =", font=(18))
lbl.place(x='10', y='510')
txt_d = Entry(tab1, width=20)
txt_d.place(x='40', y='515')
txt_d.insert(0,'3')

lbl = Label(tab1, text="Выберите метод решения: ", font=(18))
lbl.place(x='10', y='540')

Var=IntVar()
Var.set(1)
Cheb=Radiobutton(tab1, text='Метод Чебышева',variable=Var, value=1, font=(18))
Cheb.place(x='10', y='570')

lbl = Label(tab1, text="Для метода Чебышева укажите число параметров: k =", font=(18))
lbl.place(x='250', y='572')
txt_k = Entry(tab1, width=20)
txt_k.place(x='658', y='575')
txt_k.insert(0,'4')

Simple=Radiobutton(tab1, text='Метод простых итераций', variable=Var, value = 2, font=(18))
Simple.place(x='10', y='600')

lbl = Label(tab1, text="Укажите n:", font=(18))
lbl.place(x='10', y='630')
txt_n = Entry(tab1, width=20)
txt_n.place(x='100', y='635')
txt_n.insert(0,'20')

lbl = Label(tab1, text="Укажите m:", font=(18))
lbl.place(x='10', y='660')
txt_m = Entry(tab1, width=20)
txt_m.place(x='100', y='665')
txt_m.insert(0,'20')

btn_exit = Button(window, text="Выход", command=exit)
btn_exit.place(x='700', y='720')

#окно для тестовой задачи
lbl = Label(tab2, text="Введите максимальное количество итераций N_max:", font=(18))
lbl.place(x='10', y='330')
txt_N_max_2 = Entry(tab2, width=20)
txt_N_max_2.place(x='410', y='335')
txt_N_max_2.insert(0,'1000')

lbl = Label(tab2, text="Введите точность вычислений eps:", font=(18))
lbl.place(x='10', y='360')
txt_eps_2 = Entry(tab2, width=20)
txt_eps_2.place(x='280', y='365')
txt_eps_2.insert(0,'0.000001')

lbl = Label(tab2, text="Границы заданы условием, но их можно поменять", font=(18))
lbl.place(x='10', y='390')

lbl = Label(tab2, text="a =", font=(18))
lbl.place(x='10', y='420')
txt_a_2 = Entry(tab2, width=20)
txt_a_2.place(x='40', y='425')
txt_a_2.insert(0,'1')

lbl = Label(tab2, text="b =", font=(18))
lbl.place(x='10', y='450')
txt_b_2 = Entry(tab2, width=20)
txt_b_2.place(x='40', y='455')
txt_b_2.insert(0,'2')

lbl = Label(tab2, text="c =", font=(18))
lbl.place(x='10', y='480')
txt_c_2 = Entry(tab2, width=20)
txt_c_2.place(x='40', y='485')
txt_c_2.insert(0,'2')

lbl = Label(tab2, text="d =", font=(18))
lbl.place(x='10', y='510')
txt_d_2 = Entry(tab2, width=20)
txt_d_2.place(x='40', y='515')
txt_d_2.insert(0,'3')

lbl = Label(tab2, text="Выберите метод решения: ", font=(18))
lbl.place(x='10', y='540')

Var=IntVar()
Var.set(1)
Cheb=Radiobutton(tab2, text='Метод Чебышева',variable=Var, value=1, font=(18))
Cheb.place(x='10', y='570')

lbl = Label(tab2, text="Для метода Чебышева укажите число параметров: k =", font=(18))
lbl.place(x='250', y='572')
txt_k_2 = Entry(tab2, width=20)
txt_k_2.place(x='658', y='575')
txt_k_2.insert(0,'4')

Simple=Radiobutton(tab2, text='Метод простых итераций', variable=Var, value = 2, font=(18))
Simple.place(x='10', y='600')

lbl = Label(tab2, text="Укажите n:", font=(18))
lbl.place(x='10', y='630')
txt_n_2 = Entry(tab2, width=20)
txt_n_2.place(x='100', y='635')
txt_n_2.insert(0,'20')

lbl = Label(tab2, text="Укажите m:", font=(18))
lbl.place(x='10', y='660')
txt_m_2 = Entry(tab2, width=20)
txt_m_2.place(x='100', y='665')
txt_m_2.insert(0,'20')

###
tab_control.pack(expand=1, fill='both') 
window.mainloop()