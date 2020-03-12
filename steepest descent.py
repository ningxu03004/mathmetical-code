import numpy as np
from sympy import diff
from sympy import symbols
from sympy import solve
import math

x1, x2, t = symbols('x1, x2, t')


def func():
    """
    目标函数
    :param x:自变量，二维变量
    :return:因变量，标量
    """
    return x1 + 1/2*x1*x1 + 1/2*x2 + x2*x2 + 3


def grad_2d(x1, x2):
    """
    目标函数的梯度
    :param x:自变量，二维向量
    :return:梯度函数
    """
    deriv0 = 1 + x1
    deriv1 = 1/2 + 2*x2
    return np.array([deriv0, deriv1])


def zhudian(f):
    # 求得min(t)的驻点
    t_diff = diff(f)
    t_min = solve(t_diff)
    return t_min


def gradient_descent_2d(grad, cur_x=np.array([0, 0]), precision=0.0001, max_iters=100):
    f = func()
    print(f"{cur_x}作为初始值开始迭代...")
    for i in range(max_iters):
        grad_cur = grad_2d(cur_x[0], cur_x[1])
        if math.sqrt(pow(grad_cur[0], 2) + pow(grad_cur[1], 2)) > precision:
            x = cur_x-t*grad_cur
            t_func = f.subs(x1, x[0]).subs(x2, x[1])
            t_min = zhudian(t_func)
            cur_x = cur_x - t_min*grad_cur
            print(cur_x)
            print("第", i, "次迭代：x 值为 ", cur_x)
    print("局部最小值 x =", cur_x)
    return cur_x


if __name__ == '__main__':
    gradient_descent_2d(grad_2d, cur_x=np.array([0, 0]), precision=0.000001, max_iters=10000)
