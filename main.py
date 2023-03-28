import numpy as np
import matplotlib.pyplot as plt

length = 10

def do_matrix(cond):
    n = length
    log_cond = np.log(cond)
    exp_vec = np.arange(-log_cond / 4., log_cond * (n + 1) / (4 * (n - 1)), log_cond / (2. * (n - 1)))
    s = np.exp(exp_vec)
    S = np.diag(s)
    U, _ = np.linalg.qr((np.random.rand(n, n) - 5.) * 200)
    V, _ = np.linalg.qr((np.random.rand(n, n) - 5.) * 200)
    matrix = U.dot(S).dot(V.T)
    matrix = matrix.dot(matrix.T)
    return matrix


cond = [5, 6, 7, 8, 9, 10, 11, 12, 14, 15]
number = len(cond)


def change_cond():
    ten_matrix = [[] for i in range(number)]
    for i in range(number):
        ten_matrix[i] = do_matrix(cond[i])
    return ten_matrix


def do_matrix_c_and_g(a, b):
    g = [0 for j in range(length)]
    c = [[0 for j in range(length)] for i in range(length)]
    for i in range(length):
        g[i] = b[i] / a[i][i]
        for j in range(length):
            if i != j:
                c[i][j] = -a[i][j] / a[i][i]
            else:
                c[i][j] = 0
    return c, g

def method_iteration(a, b, epsilon, N_max):
    length = 10
    fun_inv_1 = do_matrix_c_and_g(a, b)
    c_end = fun_inv_1[0]
    c_norm = np.linalg.norm(c_end)
    c_cond = np.linalg.cond(c_end)
    g_end = fun_inv_1[1]
    if np.linalg.det(a) == 0 and c_cond < 1:
        return None
    iter_ = 0
    x_consider_norm = [0 for i in range(length)]
    x_norm_ = 1
    x_norm = []
    x_solve_norm = [0 for i in range(length)]
    ratio = (c_norm * epsilon) / (1 - c_norm)
    while x_norm_ > ratio and iter_ < N_max:
        x_save = x_norm_
        for i in range(length):
            summa = 0
            summa += g_end[i]
            for j in range(length):
                summa += c_end[i][j] * b[j]
            b[i] = summa
            x_solve_norm[i] = x_consider_norm[i] - b[i]
            x_consider_norm[i] = b[i]
        x_norm_ = np.linalg.norm(x_solve_norm)
        x_norm.append(x_norm_)
        iter_ += 1
        if x_norm_ == 0 or x_norm_ < epsilon:
            x_norm_ = x_save
            return x_consider_norm, iter_, x_norm, x_norm_
    return x_consider_norm, iter_, x_norm, x_norm_


cond = [5, 6, 7, 8, 9, 10, 11, 12, 14, 15]
length = 10
massive_a = change_cond()
x = [1 for j in range(length)]
massive_b = [0 for i in range(length)]
for i in range(length):
    massive_b[i] = massive_a[i].dot(x)

e_1 = 1e-16
N = 100


def loop(massive_a_, massive_b_, e1, n_max):
    x_end = [0 for i in range(length)]
    iteration = [0 for i in range(length)]
    x_normal = [0 for i in range(length)]
    for i in range(length):
        value = method_iteration(massive_a_[i], massive_b_[i], e1, n_max)
        x_end[i], iteration[i], x_normal[i] = value[0], value[1], value[3]
    return x_end, iteration, x_normal


plt.figure(1)
plt.grid()
plt.yscale('log')
plt.xlabel('число обусловленности')
plt.ylabel('относительная погрешность')
plt.title("График зависимости относительной погрешности от числа обусловленности")
plt.plot(cond, loop(massive_a, massive_b, e_1, 40)[2], label='Максимальное количество итераций = 30')
plt.plot(cond, loop(massive_a, massive_b, e_1, 30)[2], label='Максимальное количество итераций = 40')
plt.plot(cond, loop(massive_a, massive_b, e_1, 50)[2], label='Максимальное количество итераций = 50')
plt.legend()

plt.figure(2)
plt.grid()
plt.xlabel('число обусловленности')
plt.ylabel('количество итераций')
plt.title("График зависимости количества итераций от числа обусловленности")
plt.plot(cond, loop(massive_a, massive_b, 1e-16, N)[1], label='Задаваемая точность = 1e-16')
plt.plot(cond, loop(massive_a, massive_b, 1e-13, N)[1], label='Задаваемая точность = 1e-13')
plt.plot(cond, loop(massive_a, massive_b, 1e-10, N)[1], label='Задаваемая точность = 1e-10')
plt.legend()

def chart_3():
    massive_b_ = [[0 for i in range(length)] for i in range(length)]
    for i in range(length):
        for j in range(length):
            massive_b_[i][j] = (1 + i * 0.01) * massive_b[0][j]
    return massive_b_

def chart_3_plus():
    percentages = [0 for j in range(length)]
    massive_a__ = [massive_a[0] for j in range(length)]
    value = loop(massive_a__, chart_3(), 1e-16, N)[2]
    value_0 = value[0]
    for i in range(length):
        percentages[i] = abs(100 - round(100*value[i]/value_0))
    return percentages



plt.figure(3)
plt.grid()
plt.xlabel('изменение вектора свободных членов в процентах')
plt.ylabel('относительная погрешность в процентах')
plt.title("График зависимости относительной погрешности при малом изменения вектора свободных членов")
plt.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], chart_3_plus(), label='число обусловленности = 5' )
plt.legend()

def chart_4():
    help_iter = method_iteration(massive_a[0], massive_b[0], e_1, 50)[1]
    rel_errors = method_iteration(massive_a[0], massive_b[0], e_1, help_iter - 1)[2]
    return rel_errors, help_iter


val = chart_4()
help_iter_ = val[1]
iterations = np.arange(0, help_iter_-1)
rel_errors_ = val[0]

plt.figure(4)
plt.grid()
plt.yscale('log')
plt.xlabel('номер итерации')
plt.ylabel('относительная погрешность')
plt.title("График зависимости относительной погрешности от номера итерации")
plt.plot(iterations, rel_errors_, color='m')
plt.legend()


def chart_5(cond_):
    iteration = [0 for k in range(length)]
    e_1_ = 1e-5
    for i in range(length):
        iteration[i] = method_iteration(massive_a[cond_ - 5], massive_b[cond_ - 5], e_1_, 100)[1]
        e_1_ *= 0.1
    return iteration

e_1__ = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14]

plt.figure(5)
plt.grid()
plt.xscale('log')
plt.xlabel('заданная точность')
plt.ylabel('количество итераций')
plt.title("График зависимости количества итераций от заданной точности ")
plt.plot(e_1__, chart_5(5), label='Число обусловленности = 5')
plt.plot(e_1__, chart_5(10), label='Число обусловленности = 10')
plt.legend()

length = 10
e_1 = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14]
massive_a = change_cond()
x = [1 for j in range(length)]
massive_b = [0 for i in range(length)]
for i in range(length):
    massive_b[i] = massive_a[i].dot(x)

def chart_6(cond_):
    relative_errors = [0 for i in range(length)]
    e_1_ = 1e-5
    for i in range(length):
        relative_errors[i] = method_iteration(massive_a[cond_ - 5], massive_b[cond_ - 5], e_1_, 200)[3]
        e_1_ *= 0.1
    return relative_errors

plt.figure(6)
plt.grid()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('заданная точность')
plt.ylabel('относительная погрешность')
plt.title("График зависимости относительной погрешности от заданной точности")
plt.plot(e_1, chart_6(5), label='Число обусловленности = 5')
plt.plot(e_1, chart_6(10), label='Число обусловленности = 10')
plt.plot(e_1, e_1, label='биссектриса')
plt.legend()
plt.show()


