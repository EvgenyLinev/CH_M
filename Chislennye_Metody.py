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

def table():
    c = Canvas(tab1, height=300, width = 900)
    c.place(x =10, y = 280)
    c.create_rectangle(0, 0, 1600, 950)
    
    u=fh()
    Eg=[0.,0.]
    for i in range (len(u[7])-1):
        if Eg[0]<abs(u[7][i]):
            Eg[0]=abs(u[7][i])
        if Eg[1]<abs(u[8][i]):
            Eg[1]=abs(u[8][i])
    #Eg[0]=u[6][1]     
    #Eg[1]=u[7][1]  
    x=u[2][len(u[2])-1]
    lbl = Label(tab1, text="Метод при текущем шаге считает до точки, имеющей координату по оси абсцисс =",  font=(18))
    lbl.place(x='10',y='285')
    lbl = Label(tab1, text=str(x), font=(18))
    lbl.place(x='605',y='285')
    lbl = Label(tab1, text="Максимальная глобальная погрешность по w при текущем шаге:",  font=(18))
    lbl.place(x='10',y='315')
    lbl = Label(tab1, text=str(Eg[0]), font=(18))
    lbl.place(x='480',y='315')
    lbl = Label(tab1, text="Максимальная глобальная погрешность по v при текущем шаге:",  font=(18))
    lbl.place(x='10',y='345')
    lbl = Label(tab1, text=str(abs(Eg[1])), font=(18))
    lbl.place(x='480',y='345')

    lbl = Label(tab1, text="w(x) - синяя траектория",  font=(18))
    lbl.place(x='170',y='102')
    lbl = Label(tab1, text="v(x) - красная траектория",  font=(18))
    lbl.place(x='170',y='132')

    table_u = np.zeros((len(u[0]), len(u)), dtype=float).tolist()
    heads = ("n",'h',"x","u численное","v численное", "u точное", "v точное", "Eu","Ev")
    table = ttk.Treeview(tab1, show='headings', columns=heads, height=13)

    for j in range (len(u)):
        for i in range (len(u[0])):
           table_u[i][j]=u[j][i]

    table.place(x='30', y='410')
    table.heading("n", text="n", anchor='w')
    table.heading("h", text="h", anchor='w')
    table.heading("x", text="x", anchor='w')
    table.heading("u численное", text="w численное", anchor='w')
    table.heading("v численное", text="v численное", anchor='w')
    table.heading("u точное", text="w точное", anchor='w')
    table.heading("v точное", text="v точное", anchor='w')
    table.heading("Eu", text="Ew", anchor='w')
    table.heading("Ev", text="Ev", anchor='w')
    table.column('#1', width=50)
    table.column('#2', width=100)
    table.column('#3', width=150)
    table.column('#4', width=180)
    table.column('#5', width=180)
    table.column('#6', width=180)
    table.column('#7', width=180)
    table.column('#8', width=180)

    for person in table_u:
        table.insert("", END, values=person)

    scrollbary = Scrollbar(tab1, orient=VERTICAL)
    scrollbarx = Scrollbar(tab1, orient=HORIZONTAL)
    table.configure(yscrollcommand=scrollbary.set, xscrollcommand=scrollbarx.set)
    scrollbarx.configure(command=table.xview)
    scrollbary.configure(command=table.yview)
    scrollbary.place(x=1400, y=430, width=15, height=250)  

    figure=plt.figure(figsize=(7,4))   
    figure.add_subplot(111).plot(u[2], u[4], 'r-',u[2],u[3],'b-')
    chart=FigureCanvasTkAgg(figure,tab1)
    chart.get_tk_widget().place(x='800', y='0')
    plt.xlabel('x', color='gray')
    plt.ylabel('u(x)',color='gray')
    plt.grid(True)
    
def exit():
    sys. exit()    


def realu(x):
    return(-3*math.exp(-1000*x)+10*math.exp(-0.01*x))

def realv(x):
    return(3*math.exp(-1000*x)+10*math.exp(-0.01*x))

def matrix1(h):
    a=-500.005
    b=499.995   
    m = np.array([[1.-a*h/4.,-b*h/4.,-a*(1/4-3**(1/2)/6.)*h,-b*(1/4-3**(1/2)/6.)*h],
                           [-b*h/4.,1.-a*h/4.,-b*(1/4-3**(1/2)/6.)*h,-a*(1/4-3**(1/2)/6.)*h],
                           [-a*(1/4+3**(1/2)/6.)*h,-b*(1/4+3**(1/2)/6.)*h,1.-a*h/4.,-b*h/4.],
                           [-b*(1/4+3**(1/2)/6.)*h,-a*(1/4+3**(1/2)/6.)*h,-b*h/4.,1.-a*h/4.]])
    return(m)

def fh():
    h=float(txt1.get())
    eps=float(txt2.get())
    a=-500.005
    b=499.995      
    vn=[0]
    H=[h]
    n=1
    i=0
    x0=0.
    u=[7.]
    v=[13.]
    x=[0.0]
    ru=[realu(x0)]
    rv=[realv(x0)]
    Eu=[0.]
    Ev=[0.]
    u_tmp=0
    v_tmp=0
    e=10**(-9)
    p=2**5
    f=0
    
    while (u[i]>eps) | (v[i]>eps):   
        matrix = np.array([[1.-a*h/4.,-b*h/4.,-a*(1/4-3**(1/2)/6.)*h,-b*(1/4-3**(1/2)/6.)*h],
                           [-b*h/4.,1.-a*h/4.,-b*(1/4-3**(1/2)/6.)*h,-a*(1/4-3**(1/2)/6.)*h],
                           [-a*(1/4+3**(1/2)/6.)*h,-b*(1/4+3**(1/2)/6.)*h,1.-a*h/4.,-b*h/4.],
                           [-b*(1/4+3**(1/2)/6.)*h,-a*(1/4+3**(1/2)/6.)*h,-b*h/4.,1.-a*h/4.]])
        vec=np.array([a*u[i] +b*v[i],a*v[i] +b*u[i],a*u[i] +b*v[i],a*v[i] +b*u[i]])
        k=np.linalg.solve(matrix,vec)
        u_tmp=(u[i]+h/2.*(k[2]+k[0]))
        v_tmp=(v[i]+h/2.*(k[3]+k[1]))    

        kloc=np.linalg.solve(matrix1(h/2),vec)
        uloc=(u[i]+h/4.*(kloc[2]+kloc[0]))
        vloc=(v[i]+h/4.*(kloc[3]+kloc[1]))    

        vec=np.array([a*uloc +b*vloc,a*vloc +b*uloc,a*uloc +b*vloc,a*vloc +b*uloc])
        kloc=np.linalg.solve(matrix1(h/2),vec)
        uloc=(uloc+h/4.*(kloc[2]+kloc[0]))
        vloc=(vloc+h/4.*(kloc[3]+kloc[1]))    

        su=(u_tmp-uloc)/(2**4-1)
        sv=(v_tmp-vloc)/(2**4-1)
        
        s=math.sqrt(su*su+sv*sv)
        if (u_tmp>(eps*100))and(v_tmp>(eps*100)):
            if e/p<=s<=e:
                u.append(uloc)
                v.append(vloc)
                x0+=h
                H.append(h)
                f=0
            elif s<e/p:
              u.append(uloc)
              v.append(vloc)            
              x0+=h
              H.append(h)
              h*=2
              f=0
            else:
              h*=0.5
              f=1
        else:
          u.append(uloc)
          v.append(vloc)  
          x0+=h
          H.append(h)
          f=0
          h=float(txt1.get()) 
           
        if f==0:
            ru.append(realu(x0))
            rv.append(realv(x0))
            Eu.append(ru[n]-u[n])
            Ev.append(rv[n]-v[n])
            x.append(x0)
            i+=1
            n+=1   
            vn.append(i) 
    return(vn,H,x,u,v,ru,rv,Eu,Ev)
    

window = Tk()
window.title("Линев Евгений, Жесткая задача, Команда Викулова Максима")
window.geometry('1500x800')

tab_control = ttk.Notebook(window)  
tab1 = ttk.Frame(tab_control)  
tab2 = ttk.Frame(tab_control)  
tab_control.add(tab1, text='Траектория')  
tab_control.add(tab2, text='Справка')  

lbl = Label(tab1, text="Введите постоянный шаг численного интегрирования (h):", font=(18))
lbl.place(x='10', y='190')

txt1 = Entry(tab1, width=10)
txt1.place(x='430', y='193')

lbl = Label(tab1, text="Для решения используется неявный метод РК оптимального 4-го порядка", font=(18))
lbl.place(x='10',y='160')

btn = Button(tab1, text="Нажмите на кнопку, чтобы построить решение", command=table)
btn.place(x='10', y='245')

lbl = Label(tab1, text="Введите точность, с которой значение функции должно попасть в окрестность нуля:", font=(18))
lbl.place(x='10', y='220')

txt2 = Entry(tab1, width=20)
txt2.place(x='630', y='225')

btn_exit = Button(window, text="Выход", command=exit)
btn_exit.place(x='700', y='720')

lbl = Label(tab2, text="Справка:",font=(18))
lbl.place(x='10', y='10')

#lbl = Label(tab2, text="Шаг h может принимать значения от нуля до бесконечности",font=(18))
#lbl.place(x='500', y='30')

lbl = Label(tab2, text="Матрица A и функция u в данной задаче принимают значения:",font=(18))
lbl.place(x='10', y='50')

lbl = Label(tab2, text="A =",font=(18))
lbl.place(x='10', y='90')

lbl = Label(tab2, text="(", font=("Arial Bold",30))
lbl.place(x='40', y='75')

lbl = Label(tab2, text="-500.005   499.995", font=(18))
lbl.place(x='55', y='77')

lbl = Label(tab2, text="499.995   -500.005", font=(18))
lbl.place(x='55', y='100')

lbl = Label(tab2, text=")", font=("Arial Bold",30))
lbl.place(x='190', y='75')

lbl = Label(tab2, text="=",font=(18))
lbl.place(x='210', y='90')

lbl = Label(tab2, text="(", font=("Arial Bold",30))
lbl.place(x='220', y='75')

lbl = Label(tab2, text="a  b", font=(18))
lbl.place(x='235', y='77')

lbl = Label(tab2, text="b  a", font=(18))
lbl.place(x='235', y='100')

lbl = Label(tab2, text=")", font=("Arial Bold",30))
lbl.place(x='265', y='75')

lbl = Label(tab2, text=";", font=(18))
lbl.place(x='280', y='90')


lbl = Label(tab2, text="u =",font=(18))
lbl.place(x='300', y='90')

lbl = Label(tab2, text="(", font=("Arial Bold",30))
lbl.place(x='330', y='75')

lbl = Label(tab2, text="w", font=(18))
lbl.place(x='345', y='77')

lbl = Label(tab2, text="v", font=(18))
lbl.place(x='347', y='100')

lbl = Label(tab2, text=")", font=("Arial Bold",30))
lbl.place(x='360', y='75')

lbl = Label(tab2, text="=",font=(18))
lbl.place(x='380', y='90')

lbl = Label(tab2, text="(", font=("Arial Bold",30))
lbl.place(x='390', y='75')

lbl = Label(tab2, text="7", font=(18))
lbl.place(x='409', y='77')

lbl = Label(tab2, text="13", font=(18))
lbl.place(x='405', y='100')

lbl = Label(tab2, text=")", font=("Arial Bold",30))
lbl.place(x='425', y='75')


lbl = Label(tab1, text="u =",font=(18))
lbl.place(x='10', y='120')

lbl = Label(tab1, text="(", font=("Arial Bold",30))
lbl.place(x='30', y='105')

lbl = Label(tab1, text="w", font=(18))
lbl.place(x='45', y='107')

lbl = Label(tab1, text="v", font=(18))
lbl.place(x='47', y='130')

lbl = Label(tab1, text=")", font=("Arial Bold",30))
lbl.place(x='60', y='105')


lbl = Label(tab1, text="w(0)=7;", font=(18))
lbl.place(x='90',y='105')

lbl = Label(tab1, text="v(0)=13;", font=(18))
lbl.place(x='90',y='130')


lbl = Label(tab2, text="n - номер шага", font=(18))
lbl.place(x='10',y='140')

lbl = Label(tab2, text="Eu - глобальная погрешность по w", font=(18))
lbl.place(x='10',y='160')

lbl = Label(tab2, text="Ev - глобальная погрешность по v", font=(18))
lbl.place(x='10',y='180')


lbl = Label(tab2, text="Для решения используется неявный метод РК оптимального 4-го порядка", font=(18))
lbl.place(x='10',y='200')

lbl = Label(tab2, text="Для уменьшения значений глобальной погрешности нужно уменьшить шаг численного интегрирования", font=(18))
lbl.place(x='10',y='220')

lbl = Label(tab2, text="Параметр для контроля локальной погрешности равен 10^(-9)", font=(18))
lbl.place(x='10',y='240')

lbl = Label(tab2, text="Реализация включает в себя контроль шага, однако!", font=(18))
lbl.place(x='10',y='260')
lbl = Label(tab2, text="Как только заданная точность*100 превышает значения обоих компонент, шаг становится константой, равной введенному в поле значению, и остается таким до конца построения траектории", font=(18))
lbl.place(x='10',y='280')
lbl = Label(tab2, text='Благодаря этому траектория не "проскочит" точку завершения', font=(18))
lbl.place(x='10',y='300')

lbl = Label(tab2, text='Для корректного выхода из программы нажмите кнопку "выход"', font=(18))
lbl.place(x='10',y='360')

c = Canvas(tab1, width=15, height=100)
c.place(x='0', y='0')

c.create_line(10, 2, 15, 2)
c.create_line(10, 2, 10, 45)
c.create_line(10, 45, 5, 50)
c.create_line(5, 50, 10, 55)
c.create_line(10, 55, 10, 99)
c.create_line(10, 99, 15, 99)

lbl = Label(tab1, text="du/dx = Au", font=(18))
lbl.place(x='20',y='10')
lbl = Label(tab1, text="u(0) = u0", font=(18))
lbl.place(x='20',y='60')

tab_control.pack(expand=1, fill='both') 

window.mainloop()