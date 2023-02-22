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


def ralgb5(calcfg, x0, tolx=1e-12, tolg=1e-12, tolf=1e-12, maxiter=2000, alpha=2.3, nsims=30, h0=1, nh=3, q1=0.9, q2=1.1):
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



class Accelerated_ellipsoid_method:
    def calc(self, calcfg, x0, maxiter=100, tolg=1E-7):
        n = len(x0)
        B_k = np.eye(n)
        r_0 = 10
        x_k = x0
        for i in range(maxiter):
            f, g = calcfg(x_k)
            norm_g = np.linalg.norm(g)
            if norm_g < tolg:
                ccode = 2
                return ccode

            Bg = B_k.T @ g
            xsi_k = Bg / np.linalg.norm(Bg)

            if (i == 0):
                h_k = 1 / (n + 1) * r_0
                beta_k = np.sqrt((n - 1) / (n + 1))
            else:
                xsi_dot = xsi_k.dot(xsi_old)
                if (h_k / beta_k < - r_k * xsi_dot):
                    if (xsi_dot ** 2 - 1) < 1E-10:
                        Hk = - h_k / beta_k * xsi_dot
                    else:
                        Hk = - h_k / beta_k * xsi_dot + np.sqrt(r_k ** 2 - (h_k / beta_k) ** 2) * np.sqrt(
                            1 - xsi_dot ** 2)
                else:
                    Hk = r_k
                h_k = self.__get_beta(r_k, Hk, n)
                beta_k = self.__get_beta(r_k, Hk, n)

            x_k = x_k - h_k * B_k @ xsi_k
            R = np.eye(n) - (beta_k - 1) * xsi_k @ xsi_k.T
            B_k = B_k * R
            if (i == 0):
                r_k = r_0 * n / np.sqrt(n ** 2 - 1)
            else:
                r_k = np.sqrt(r_k ** 2 - (Hk / 2) ** 2 * (1 - beta_k ** 2) ** 2 / beta_k ** 2)

            print(x_k)
            xsi_old = xsi_k

    def __get_beta(self, r, H, n):
        return np.sqrt(
            np.sqrt((n - 1) * (n + 1) + ((2 * r ** 2 - H ** 2) / ((n + 1) * H ** 2)) ** 2) - (2 * r ** 2 - H ** 2) / (
                        (n + 1) * H ** 2))

    def __get_h(self, r, H, n):
        return 0.5 * (H - np.sqrt((n - 1) * (n + 1) * H ** 2 + ((2 * r ** 2 - H ** 2) / ((n + 1) * H ** 2)) ** 2) + (
                    (2 * r ** 2 - H ** 2) / ((n + 1) * H ** 2)))


class EMShor:
    def calc(self,calcfg, x0, tolx=1e-12, tolg=1e-12, tolf=1e-12, maxiter= 2000, rad = 5000, intp = 10):
        dn = float(len(x0))
        beta = math.sqrt((dn - 1) / (dn + 1))  # row02
        x = x0
        radn = rad
        B = np.eye(len(x))
        x_old = np.copy(x0)
        f_old, g1 = calcfg(x)
        for itn in range(maxiter):
            f, g1 = calcfg(x)
            g = B.T @ g1
            dg = np.linalg.norm(g)
            if (radn * dg < tolg):
                ccode = 2
                return x, f, itn, ccode
            xi = (1 / dg) * g
            dx = B @ xi
            hs = radn / (dn + 1)
            x -= hs * dx
            B += (beta - 1) * B * xi * xi.T
            radn = radn / np.sqrt(1 - 1 / dn) / np.sqrt(1 + 1 / dn)

            if np.linalg.norm(hs * dx) < tolx:
                ccode = 3
                return x, f, itn, ccode
        ccode = 4
        return x, f, itn, ccode