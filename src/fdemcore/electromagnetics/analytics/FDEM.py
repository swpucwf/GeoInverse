"""FDEM 解析解（全空间偶极子、层状介质 Hz）。

从 SimPEG electromagnetics/analytics/FDEM.py 适配，
将 ``from simpeg import utils`` 改为 ``from discretize.utils``。

原始实现：Ward & Hohmann (1988) Eq. 4.56 / Eq. 2.57。

Copyright SimPEG Project (MIT); 适配 Copyright 2026 LWD-EM-Inv Team.
"""
import numpy as np
from scipy.constants import mu_0, pi, epsilon_0
from discretize.utils import mkvc, as_array_n_by_dim


def hzAnalyticDipoleF(r, freq, sigma, secondary=True, mu=mu_0):
    """层状半空间磁偶极子 Hz 解析解 (Ward & Hohmann, 1988, Eq. 4.56)。

    Parameters
    ----------
    r : float or array
        收发距 (m)。
    freq : float or array
        频率 (Hz)。
    sigma : float
        均匀半空间电导率 (S/m)。
    secondary : bool
        True 返回二次场（减掉一次场），False 返回总场。
    mu : float
        磁导率 (H/m)，默认 mu_0。

    Returns
    -------
    hz : complex array
        Hz 响应 (A/m)。
    """
    r = np.abs(r)
    k = np.sqrt(-1j * 2.0 * np.pi * freq * mu * sigma)

    m = 1
    front = m / (2.0 * np.pi * (k**2) * (r**5))
    back = 9 - (
        9 + 9j * k * r - 4 * (k**2) * (r**2) - 1j * (k**3) * (r**3)
    ) * np.exp(-1j * k * r)
    hz = front * back

    if secondary:
        hp = -1 / (4 * np.pi * r**3)
        hz = hz - hp

    if hz.ndim == 1:
        hz = mkvc(hz, 2)

    return hz


def MagneticDipoleWholeSpace(
    XYZ, srcLoc, sig, f, moment, fieldType="b", mu_r=1, eps_r=1, **kwargs
):
    """全空间磁偶极子解析解 (Ward & Hohmann, 1988, Eq. 2.57)。

    Parameters
    ----------
    XYZ : array_like
        观测点坐标 (n, 3) 或 (3,)。
    srcLoc : array_like
        源位置 (3,)。
    sig : float
        电导率 (S/m)。
    f : float or array
        频率 (Hz)。
    moment : str or tuple
        偶极矩方向，'X'/'Y'/'Z' 或 (mx, my, mz)。
    fieldType : str
        场量类型，'e'/'h'/'b'。
    mu_r : float
        相对磁导率。
    eps_r : float
        相对介电常数。

    Returns
    -------
    Fx, Fy, Fz : complex arrays
        三个分量的场值。
    """
    orient = kwargs.pop("orientation", None)
    if orient is not None:
        raise TypeError(
            "orientation kwarg has been removed, please use the moment argument",
        )
        magnitude = moment
        moment = orient
    else:
        magnitude = 1
    mu = kwargs.pop("mu", None)
    if mu is not None:
        raise TypeError("mu kwarg has been removed, please use the mu_r argument.")
        mu_r = mu / mu_0

    mu = mu_0 * mu_r
    eps = epsilon_0 * eps_r
    w = 2 * np.pi * f

    if isinstance(moment, str):
        if moment == "X":
            mx, my, mz = 1.0, 0.0, 0.0
        elif moment == "Y":
            mx, my, mz = 0.0, 1.0, 0.0
        elif moment == "Z":
            mx, my, mz = 0.0, 0.0, 1.0
        else:
            raise NotImplementedError("String type for moment not recognized")
        mx, my, mz = mx * magnitude, my * magnitude, mz * magnitude
    else:
        mx, my, mz = moment[0], moment[1], moment[2]

    XYZ = as_array_n_by_dim(XYZ, 3)

    dx = XYZ[:, 0] - srcLoc[0]
    dy = XYZ[:, 1] - srcLoc[1]
    dz = XYZ[:, 2] - srcLoc[2]

    r = np.sqrt(dx**2.0 + dy**2.0 + dz**2.0)
    k = np.sqrt(-1j * w * mu * sig + w**2 * mu * eps)
    kr = k * r

    if fieldType in ["h", "b"]:
        front = 1 / (4.0 * pi * r**3.0) * np.exp(-1j * kr)
        mid = -(kr**2.0) + 3.0 * 1j * kr + 3.0

        Fx = front * (
            mx * ((dx / r) ** 2.0 * mid + (kr**2.0 - 1j * kr - 1.0))
            + my * ((dy * dx / r**2.0) * mid)
            + mz * ((dx * dz / r**2.0) * mid)
        )

        Fy = front * (
            mx * ((dx * dy / r**2.0) * mid)
            + my * ((dy / r) ** 2.0 * mid + (kr**2.0 - 1j * kr - 1.0))
            + mz * ((dy * dz / r**2.0) * mid)
        )

        Fz = front * (
            mx * ((dx * dz / r**2.0) * mid)
            + my * ((dy * dz / r**2.0) * mid)
            + mz * ((dz / r) ** 2.0 * mid + (kr**2.0 - 1j * kr - 1.0))
        )

        if fieldType == "b":
            Fx, Fy, Fz = mu * Fx, mu * Fy, mu * Fz

    elif fieldType == "e":
        front = 1j * w * mu * (1 + 1j * kr) / (4.0 * pi * r**3.0) * np.exp(-1j * kr)

        Fx = front * (my * (dz / r) + mz * (-dy / r))

        Fy = front * (mx * (-dz / r) + mz * (dx / r))

        Fz = front * (mx * (dy / r) + my * (-dx / r))

    return Fx, Fy, Fz


def ElectricDipoleWholeSpace(
    XYZ, srcLoc, sig, f, moment="X", fieldType="e", mu_r=1, eps_r=1, **kwargs
):
    """全空间电偶极子解析解。

    Parameters
    ----------
    XYZ : array_like
        观测点坐标 (n, 3) 或 (3,)。
    srcLoc : array_like
        源位置 (3,)。
    sig : float
        电导率 (S/m)。
    f : float or array
        频率 (Hz)。
    moment : str or tuple
        偶极矩方向，'X'/'Y'/'Z' 或 (mx, my, mz)。
    fieldType : str
        场量类型，'e'/'h'/'b'。
    mu_r : float
        相对磁导率。
    eps_r : float
        相对介电常数。

    Returns
    -------
    Fx, Fy, Fz : complex arrays
        三个分量的场值。
    """
    orient = kwargs.pop("orientation", None)
    if orient is not None:
        raise TypeError(
            "orientation kwarg has been removed, please use the moment argument."
        )
    mu = kwargs.pop("mu", None)
    if mu is not None:
        raise TypeError("mu kwarg has been removed, please use the mu_r argument.")
    cur = kwargs.pop("current", None)
    if cur is not None:
        raise TypeError(
            "current kwarg has been removed, please use the moment argument.",
        )
    else:
        magnitude = 1
    length = kwargs.pop("length", None)
    if length is not None:
        raise TypeError(
            "length kwarg has been removed, please use the moment argument."
        )

    mu = mu_0 * mu_r
    eps = epsilon_0 * eps_r
    w = 2 * np.pi * f

    if isinstance(moment, str):
        if moment.upper() == "X":
            mx, my, mz = 1.0, 0.0, 0.0
        elif moment.upper() == "Y":
            mx, my, mz = 0.0, 1.0, 0.0
        elif moment.upper() == "Z":
            mx, my, mz = 0.0, 0.0, 1.0
        else:
            raise NotImplementedError("String type for moment not recognized")
        mx, my, mz = mx * magnitude, my * magnitude, mz * magnitude

    else:
        mx, my, mz = moment[0], moment[1], moment[2]

    XYZ = as_array_n_by_dim(XYZ, 3)

    dx = XYZ[:, 0] - srcLoc[0]
    dy = XYZ[:, 1] - srcLoc[1]
    dz = XYZ[:, 2] - srcLoc[2]

    r = np.sqrt(dx**2.0 + dy**2.0 + dz**2.0)
    k = np.sqrt(-1j * w * mu * sig + w**2 * mu * eps)
    kr = k * r

    if fieldType == "e":
        front = 1 / (4.0 * np.pi * sig * r**3) * np.exp(-1j * k * r)
        mid = -(k**2) * r**2 + 3 * 1j * k * r + 3

        Fx = front * (
            mx * ((dx**2 / r**2) * mid + (k**2 * r**2 - 1j * k * r - 1.0))
            + my * (dy * dx / r**2) * mid
            + mz * (dz * dx / r**2) * mid
        )

        Fy = front * (
            mx * (dx * dy / r**2) * mid
            + my * ((dy**2 / r**2) * mid + (k**2 * r**2 - 1j * k * r - 1.0))
            + mz * (dz * dy / r**2) * mid
        )

        Fz = front * (
            mx * (dx * dz / r**2) * mid
            + my * (dy * dz / r**2) * mid
            + mz * ((dz**2 / r**2) * mid + (k**2 * r**2 - 1j * k * r - 1.0))
        )

    elif fieldType in ["h", "b"]:
        front = (1 + 1j * kr) / (4.0 * np.pi * r**2) * np.exp(-1j * k * r)

        Fx = front * (my * (dz / r) + mz * (-dy / r))

        Fy = front * (mx * (-dz / r) + mz * (dx / r))

        Fz = front * (mx * (dy / r) + my * (-dx / r))

        if fieldType == "b":
            Fx, Fy, Fz = mu * Fx, mu * Fy, mu * Fz

    return Fx, Fy, Fz
