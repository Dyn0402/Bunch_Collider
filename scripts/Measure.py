#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 04 2:58 PM 2020
Created in PyCharm
Created as QGP_Scripts/Measure.py

@author: Dylan Neff, dylan

Updated version copied from UCLA_Nuclear on 3/16/21
"""

import math

import numpy as np


class Measure:
    def __init__(self, val=0, err=0):
        self._val = val
        self._err = err

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, val):
        self._val = val

    @val.deleter
    def val(self):
        del self._val

    @property
    def err(self):
        return self._err

    @err.setter
    def err(self, err):
        self._err = err

    @err.deleter
    def err(self):
        del self._err

    def str_latex(self):
        return str(self).replace(' ± ', ' \\pm ')

    def __neg__(self):
        return Measure(-self.val, self.err)

    def __add__(self, o):
        result = Measure()
        if type(o) == Measure:
            result.val = self.val + o.val
            result.err = (self.err**2 + o.err**2)**0.5
        else:
            try:
                c = float(o)
                result.val = self.val + c
                result.err = self.err
            except ValueError:
                result = NotImplemented
        return result

    def __radd__(self, o):
        return self.__add__(o)

    def __sub__(self, o):
        result = Measure()
        if type(o) == Measure:
            result.val = self.val - o.val
            result.err = (self.err**2 + o.err**2)**0.5
        else:
            try:
                c = float(o)
                result.val = self.val - c
                result.err = self.err
            except ValueError:
                result = NotImplemented
        return result

    def __rsub__(self, o):
        return -self.__sub__(o)

    def __mul__(self, o):  # What if self.val or o.val == 0?
        result = Measure()
        if type(o) == Measure:
            result.val = self.val * o.val
            result.err = ((self.err * o.val) ** 2 + (o.err * self.val) ** 2) ** 0.5
        else:
            try:
                c = float(o)
                result.val = self.val * c
                result.err = abs(c) * self.err
            except ValueError:
                result = NotImplemented
        return result

    def __rmul__(self, o):
        return self.__mul__(o)

    def __truediv__(self, o):
        if o == 0:
            return Measure(float('nan'), float('nan'))
        result = Measure()
        if type(o) == Measure:
            result.val = self.val / o.val
            result.err = ((self.err / o.val) ** 2 + (o.err * self.val / o.val ** 2) ** 2) ** 0.5
        else:
            try:
                c = float(o)
                result.val = self.val / c
                result.err = self.err / abs(c)
            except ValueError:
                result = NotImplemented
        return result

    def __rtruediv__(self, o):  # Might lead to an error when inverting with zeros, maybe better to write out explicitly
        return self.__truediv__(o)**-1

    def __pow__(self, o):
        result = Measure()
        if type(o) == Measure:
            result.val = self.val ** o.val
            result.err = abs(result.val) * ((o.val / self.val * self.err)**2 + (math.log(self.val) * o.err)**2)**0.5
        else:
            try:
                c = float(o)
                result.val = self.val ** c
                result.err = abs(result.val * c * self.err / self.val)
            except ValueError:
                result = NotImplemented
        return result

    def sqrt(self):
        return self**0.5

    def __rpow__(self, o):  # If o is Measure then o.__pow__ should run, so assume o is not Measure
        result = Measure()
        try:
            c = float(o)
            result.val = c**self.val
            result.err = abs(result.val * math.log(c) * self.err)
        except ValueError:
            result = NotImplemented
        return result

    def __abs__(self):
        return Measure(abs(self.val), self.err)

    def __eq__(self, o):
        if type(o) == Measure:
            return self.val == o.val
        try:
            return self.val == float(o)
        except ValueError:
            return NotImplemented

    def __ne__(self, o):
        if type(o) == Measure:
            return self.val != o.val
        try:
            return self.val != float(o)
        except ValueError:
            return NotImplemented

    def __lt__(self, o):
        if type(o) == Measure:
            return self.val < o.val
        try:
            return self.val < float(o)
        except ValueError:
            return NotImplemented

    def __le__(self, o):
        if type(o) == Measure:
            return self.val <= o.val
        try:
            return self.val <= float(o)
        except ValueError:
            return NotImplemented

    def __gt__(self, o):
        if type(o) == Measure:
            return self.val > o.val
        try:
            return self.val > float(o)
        except ValueError:
            return NotImplemented

    def __ge__(self, o):
        if type(o) == Measure:
            return self.val >= o.val
        try:
            return self.val >= float(o)
        except ValueError:
            return NotImplemented

    def conjugate(self):
        return self

    def __str__(self):
        dec = err_dec(self.err) if self.err != 0 else err_dec(self.val, 5)
        f_or_e = float_or_exp(self.val, dec)
        if f_or_e == 'e' and np.isfinite(self.err):
            try:
                precision = 1 + math.floor(math.log10(abs(self.val / self.err))) if self.err != 0 else 2
            except ValueError:
                precision = 2
            precision = max(precision, 2)
            val, err = match_exponents(self.val, self.err, precision)
            e_str = f'{val} ± {err}'
            f_str = f'{self.val:.{dec}f} ± {self.err:.{dec}f}'
            if len(e_str) < len(f_str):
                return e_str
        return f'{self.val:.{dec}f} ± {self.err:.{dec}f}'

    def __repr__(self):
        return str(self)


def err_dec(x, prec=2):
    if math.isinf(x) or math.isnan(x):
        return 0
    dec = 0
    while int(abs(x)) < 10**(prec - 1) and x != 0:
        x *= 10
        dec += 1
    return dec


def float_or_exp(x, dec, len_thresh=7):
    """
    Decide whether to return a float or scientific notation string based on the value and decimal precision.
    :param x: Value to decide on.
    :param dec: Number of decimal places to round to.
    :param len_thresh: Threshold for switching to scientific notation.
    :return:
    """
    if len(f'{x:.{dec}f}') > len_thresh:
        return 'e'
    else:
        return 'f'


def get_exponent(value):
    # Get the exponent of the number in scientific notation
    if value == 0:
        return 0
    exponent = int(math.floor(math.log10(abs(value))))
    return exponent


def match_exponents(value1, value2, precision=2):
    # Get the exponent of the first value
    exponent = get_exponent(value1)

    # Adjust both values to have the same exponent
    adjusted_value1 = value1 / (10 ** exponent)
    adjusted_value2 = value2 / (10 ** exponent)

    # Format both values with the same exponent
    formatted_str1 = f"{adjusted_value1:.{precision}f}e{exponent:+}"
    formatted_str2 = f"{adjusted_value2:.{precision}f}e{exponent:+}"

    return formatted_str1, formatted_str2


def log(x, base=math.e):
    try:
        float(base)
    except ValueError:
        return NotImplemented

    if type(x) == Measure:
        val = math.log(x.val, base)
        err = abs(x.err / x.val)
        return Measure(val, err)
    else:
        try:
            return math.log(float(x), base)
        except ValueError:
            return NotImplemented
