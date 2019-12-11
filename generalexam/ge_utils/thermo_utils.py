"""Thermodynamic conversions.

This file is a subset of the following:

https://github.com/sharppy/SHARPpy/blob/master/sharppy/sharptab/thermo.py
"""

import numpy

ROCP = 0.28571426       # R over Cp
ZEROCNK = 273.15        # Zero Celsius in Kelvins

c1 = 0.0498646455
c2 = 2.4082965
c3 = 7.07475
c4 = 38.9114
c5 = 0.0915
c6 = 1.2035
eps = 0.62197


def drylift(p, t, td):
    '''
    Lifts a parcel to the LCL and returns its new level and temperature.
    Parameters
    ----------
    p : number, numpy array
        Pressure of initial parcel in hPa
    t : number, numpy array
        Temperature of inital parcel in C
    td : number, numpy array
        Dew Point of initial parcel in C
    Returns
    -------
    p2 : number, numpy array
        LCL pressure in hPa
    t2 : number, numpy array
        LCL Temperature in C
    '''
    t2 = lcltemp(t, td)
    p2 = thalvl(theta(p, t, 1000.), t2)
    return p2, t2


def lcltemp(t, td):
    '''
    Returns the temperature (C) of a parcel when raised to its LCL.
    Parameters
    ----------
    t : number, numpy array
        Temperature of the parcel (C)
    td : number, numpy array
        Dewpoint temperature of the parcel (C)
    Returns
    -------
    Temperature (C) of the parcel at it's LCL.
    '''
    s = t - td
    dlt = s * (1.2185 + 0.001278 * t + s * (-0.00219 + 1.173e-5 * s -
                                            0.0000052 * t))
    return t - dlt


def thalvl(theta, t):
    '''
    Returns the level (hPa) of a parcel.
    Parameters
    ----------
    theta : number, numpy array
        Potential temperature of the parcel (C)
    t : number, numpy array
        Temperature of the parcel (C)
    Returns
    -------
    Pressure Level (hPa [float]) of the parcel
    '''

    t = t + ZEROCNK
    theta = theta + ZEROCNK
    return 1000. / (numpy.power((theta / t),(1./ROCP)))


def theta(p, t, p2=1000.):
    '''
    Returns the potential temperature (C) of a parcel.
    Parameters
    ----------
    p : number, numpy array
        The pressure of the parcel (hPa)
    t : number, numpy array
        Temperature of the parcel (C)
    p2 : number, numpy array (default 1000.)
        Reference pressure level (hPa)
    Returns
    -------
    Potential temperature (C)
    '''
    return ((t + ZEROCNK) * numpy.power((p2 / p),ROCP)) - ZEROCNK


def wobf(t):
    '''
    Implementation of the Wobus Function for computing the moist adiabats.
    Parameters
    ----------
    t : number, numpy array
        Temperature (C)
    Returns
    -------
    Correction to theta (C) for calculation of saturated potential temperature.
    '''
    t = t - 20
    if type(t) == type(numpy.array([])) or type(t) == type(numpy.ma.array([])):
        npol = 1. + t * (-8.841660499999999e-3 + t * ( 1.4714143e-4 + t * (-9.671989000000001e-7 + t * (-3.2607217e-8 + t * (-3.8598073e-10)))))
        npol = 15.13 / (numpy.power(npol,4))
        ppol = t * (4.9618922e-07 + t * (-6.1059365e-09 + t * (3.9401551e-11 + t * (-1.2588129e-13 + t * (1.6688280e-16)))))
        ppol = 1 + t * (3.6182989e-03 + t * (-1.3603273e-05 + ppol))
        ppol = (29.93 / numpy.power(ppol,4)) + (0.96 * t) - 14.8
        correction = numpy.zeros(t.shape, dtype=numpy.float64)
        correction[t <= 0] = npol[t <= 0]
        correction[t > 0] = ppol[t > 0]
        return correction
    else:
        if t is numpy.ma.masked:
            return t
        if t <= 0:
            npol = 1. + t * (-8.841660499999999e-3 + t * ( 1.4714143e-4 + t * (-9.671989000000001e-7 + t * (-3.2607217e-8 + t * (-3.8598073e-10)))))
            npol = 15.13 / (numpy.power(npol,4))
            return npol
        else:
            ppol = t * (4.9618922e-07 + t * (-6.1059365e-09 + t * (3.9401551e-11 + t * (-1.2588129e-13 + t * (1.6688280e-16)))))
            ppol = 1 + t * (3.6182989e-03 + t * (-1.3603273e-05 + ppol))
            ppol = (29.93 / numpy.power(ppol,4)) + (0.96 * t) - 14.8
            return ppol



def satlift(p, thetam):
    '''
    Returns the temperature (C) of a saturated parcel (thm) when lifted to a
    new pressure level (hPa)
    Parameters
    ----------
    p : number
        Pressure to which parcel is raised (hPa)
    thetam : number
        Saturated Potential Temperature of parcel (C)
    Returns
    -------
    Temperature (C) of saturated parcel at new level
    '''
    #if type(p) == type(numpy.array([p])) or type(thetam) == type(numpy.array([thetam])):
    if numpy.fabs(p - 1000.) - 0.001 <= 0: return thetam
    eor = 999
    while numpy.fabs(eor) - 0.1 > 0:
        if eor == 999:                  # First Pass
            pwrp = numpy.power((p / 1000.),ROCP)
            t1 = (thetam + ZEROCNK) * pwrp - ZEROCNK
            e1 = wobf(t1) - wobf(thetam)
            rate = 1
        else:                           # Successive Passes
            rate = (t2 - t1) / (e2 - e1)
            t1 = t2
            e1 = e2
        t2 = t1 - (e1 * rate)
        e2 = (t2 + ZEROCNK) / pwrp - ZEROCNK
        e2 += wobf(t2) - wobf(e2) - thetam
        eor = e2 * rate
    return t2 - eor


def wetlift(p, t, p2):
    '''
    Lifts a parcel moist adiabatically to its new level.
    Parameters
    -----------
    p : number
        Pressure of initial parcel (hPa)
    t : number
        Temperature of initial parcel (C)
    p2 : number
        Pressure of final level (hPa)
    Returns
    -------
    Temperature (C)
    '''
    thta = theta(p, t, 1000.)
    if thta is numpy.ma.masked or p2 is numpy.ma.masked:
        return numpy.ma.masked
    thetam = thta - wobf(thta) + wobf(t)
    return satlift(p2, thetam)


def wetbulb(p, t, td):
    '''
    Calculates the wetbulb temperature (C) for the given parcel
    Parameters
    ----------
    p : number
        Pressure of parcel (hPa)
    t : number
        Temperature of parcel (C)
    td : number
        Dew Point of parcel (C)
    Returns
    -------
    Wetbulb temperature (C)
    '''
    p2, t2 = drylift(p, t, td)
    return wetlift(p2, t2, p)
