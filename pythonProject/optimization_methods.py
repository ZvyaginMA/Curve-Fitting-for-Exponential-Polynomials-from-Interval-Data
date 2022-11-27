import numpy as np
import math


def subGradDes(calcfg, x_0, rad_search, maxiter=100, tolx=1e-12, tolg=1e-12, tolf=1e-12, print_steps=-1, stop_crit= ["f", "g", "x"]):
    """
    exit_code =
    0 - старт программы
    1 - остановка по f
    2 - остановка по g
    3 - остановка по x
    4 - остановка по iter
    """
    exit_code = 0
    x_all = []
    f_all = []
    norm_gk = []
    alpha = 1
    f_k, g_k = calcfg(x_0)
    x_all.append(x_0)
    f_all.append(f_k)

    h_k = 1
    h_all = [h_k]
    if print_steps >= 0:
        print("k, f_k, g_k , h_k, x_new ")
    for k in range(1, maxiter):
        g_norm = np.linalg.norm(g_k)
        norm_gk.append(np.linalg.norm(g_k))
        if "g" in stop_crit:
            if g_norm < tolg:
                exit_code = 2
                break
        h_k = rad_search / (2 * g_norm * math.sqrt(k)) * alpha
        x_new = 0.5 * ((x_all[-1] - h_k * g_k) + np.abs(x_all[-1] - h_k * g_k))
        h_all.append(h_k)
        f_k, g_new = calcfg(x_new)
        if g_new @ g_k / np.linalg.norm(g_new) / g_norm < -0.7:
            alpha *= 0.9

        g_k = g_new
        if print_steps >= 0:
            if k % print_steps == 0:
                print(k, f_k, g_k, h_k, x_new)
        x_all.append(x_new)
        f_all.append(f_k)
        if "f" in stop_crit:
            if abs(f_all[-1] - f_all[-2]) < tolf:
                exit_code = 1
                break

        if "x" in stop_crit:
            if np.linalg.norm(x_all[-1] - x_all[-2]) < tolx:
                exit_code = 3
                break
    if exit_code == 0:
        exit_code = 4
    return np.array(x_all), np.array(f_all),np.array(h_all), {"exit_code" : exit_code, "iter" : k}




def ralgb5_with_proj(calcfg, x0, tol=1e-12, maxiter=2000, alpha=2.3, nsims=30, h0=1, nh=3, q1=0.9, q2=1.1, tolx=1e-12, tolg=1e-12, tolf=1e-12):
    """
        exit_code =
        0 - старт программы
        1 - остановка по f
        2 - остановка по g
        3 - остановка по x
        4 - остановка по iter
    """
    m = len(x0)
    hs = h0
    B = np.eye(m)
    vf = np.zeros(nsims) + float('inf')
    w = 1./alpha - 1

    x = np.copy(x0)
    xr = np.copy(x0)

    nit = 0
    ncalls = 1
    fr, g0 = calcfg(xr)

    if np.linalg.norm(g0) < tolg:
        ccode = 2
        return xr, fr, nit, ncalls, ccode

    while nit <= maxiter:
        vf[nsims-1] = fr

        g1 = B.T @ g0
        dx = B @ (g1 / np.linalg.norm(g1))
        normdx = np.linalg.norm(dx)

        d = 1
        cal = 0
        deltax = 0

        while d > 0 and cal <= 500:
            x = 0.5 * (x - hs*dx + np.abs(x - hs*dx))
            deltax = deltax + hs * normdx

            ncalls += 1
            f, g1 = calcfg(x)

            if f < fr:
                fr = f
                xr = x

            if np.linalg.norm(g1) < tolg:
                ccode = 2
                return xr, fr, nit, ncalls, ccode

            if np.mod(cal, nh) == 0:
                hs = hs * q2
            d = dx @ g1

            cal += 1

        if cal > 500:
            ccode = 5
            return xr, fr, nit, ncalls, ccode

        if cal == 1:
            hs = hs * q1

        if deltax < tolx:
            ccode = 3
            return xr, fr, nit, ncalls, ccode

        dg = B.T @ (g1 - g0)
        xi = dg / np.linalg.norm(dg)

        B = B + w * np.outer((B @ xi), xi)
        g0 = g1

        vf = np.roll(vf, 1)
        vf[0] = abs(fr - vf[0])

        if abs(fr) > 1:
            deltaf = np.sum(vf)/abs(fr)
        else:
            deltaf = np.sum(vf)

        if deltaf < tolf:
            ccode = 1
            return xr, fr, nit, ncalls, ccode

        nit += 1

    ccode=4
    return xr, fr, nit, ncalls, ccode


def ralgb5(calcfg, x0, tol=1e-12, maxiter=2000, alpha=2.3, nsims=30, h0=1, nh=3, q1=0.9, q2=1.1, tolx=1e-12, tolg=1e-12, tolf=1e-12):
    """
        exit_code =
        0 - старт программы
        1 - остановка по f
        2 - остановка по g
        3 - остановка по x
        4 - остановка по iter
    """
    m = len(x0)
    hs = h0
    B = np.eye(m)
    vf = np.zeros(nsims) + float('inf')
    w = 1./alpha - 1

    x = np.copy(x0)
    xr = np.copy(x0)

    nit = 0
    ncalls = 1
    fr, g0 = calcfg(xr)

    if np.linalg.norm(g0) < tolg:
        ccode = 2
        return xr, fr, nit, ncalls, ccode

    while nit <= maxiter:
        vf[nsims-1] = fr

        g1 = B.T @ g0
        dx = B @ (g1 / np.linalg.norm(g1))
        normdx = np.linalg.norm(dx)

        d = 1
        cal = 0
        deltax = 0

        while d > 0 and cal <= 500:
            x = x - hs*dx
            deltax = deltax + hs * normdx

            ncalls += 1
            f, g1 = calcfg(x)

            if f < fr:
                fr = f
                xr = x

            if np.linalg.norm(g1) < tolg:
                ccode = 2
                return xr, fr, nit, ncalls, ccode

            if np.mod(cal, nh) == 0:
                hs = hs * q2
            d = dx @ g1

            cal += 1

        if cal > 500:
            ccode = 5
            return xr, fr, nit, ncalls, ccode

        if cal == 1:
            hs = hs * q1

        if deltax < tolx:
            ccode = 3
            return xr, fr, nit, ncalls, ccode

        dg = B.T @ (g1 - g0)
        xi = dg / np.linalg.norm(dg)

        B = B + w * np.outer((B @ xi), xi)
        g0 = g1

        vf = np.roll(vf, 1)
        vf[0] = abs(fr - vf[0])

        if abs(fr) > 1:
            deltaf = np.sum(vf)/abs(fr)
        else:
            deltaf = np.sum(vf)

        if deltaf < tolf:
            ccode = 1
            return xr, fr, nit, ncalls, ccode

        nit += 1

    ccode=4
    return xr, fr, nit, ncalls, ccode