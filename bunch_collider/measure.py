"""
Value-with-uncertainty arithmetic.

``Measure`` wraps a (value, error) pair and propagates errors through the
standard arithmetic operations (+, -, *, /, **) using linear error propagation.
It also formats itself as ``"value ± error"`` with automatic decimal-place
selection, switching to scientific notation when the string would otherwise be
too long.

Example
-------
>>> from bunch_collider.measure import Measure
>>> a = Measure(3.14159, 0.00023)
>>> b = Measure(2.71828, 0.00015)
>>> print(a + b)
5.85987 ± 0.00028
>>> print(a * b)
8.5397 ± 0.0010
"""

import math
import numpy as np


class Measure:
    """
    A scalar measurement with an associated uncertainty.

    Arithmetic operations propagate errors assuming uncorrelated Gaussian
    uncertainties via standard linear (first-order) error propagation.

    Parameters
    ----------
    val : float
        Central value of the measurement.
    err : float
        One-sigma uncertainty.
    """

    def __init__(self, val=0, err=0):
        self._val = val
        self._err = err

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def val(self):
        """Central value."""
        return self._val

    @val.setter
    def val(self, val):
        self._val = val

    @val.deleter
    def val(self):
        del self._val

    @property
    def err(self):
        """One-sigma uncertainty."""
        return self._err

    @err.setter
    def err(self, err):
        self._err = err

    @err.deleter
    def err(self):
        del self._err

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def str_latex(self):
        """Return a LaTeX-formatted string (``value \\pm error``)."""
        return str(self).replace(' ± ', r' \pm ')

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __neg__(self):
        return Measure(-self.val, self.err)

    def __add__(self, o):
        result = Measure()
        if isinstance(o, Measure):
            result.val = self.val + o.val
            result.err = (self.err ** 2 + o.err ** 2) ** 0.5
        else:
            try:
                c = float(o)
                result.val = self.val + c
                result.err = self.err
            except (ValueError, TypeError):
                return NotImplemented
        return result

    def __radd__(self, o):
        return self.__add__(o)

    def __sub__(self, o):
        result = Measure()
        if isinstance(o, Measure):
            result.val = self.val - o.val
            result.err = (self.err ** 2 + o.err ** 2) ** 0.5
        else:
            try:
                c = float(o)
                result.val = self.val - c
                result.err = self.err
            except (ValueError, TypeError):
                return NotImplemented
        return result

    def __rsub__(self, o):
        return (-self).__add__(o)

    def __mul__(self, o):
        result = Measure()
        if isinstance(o, Measure):
            result.val = self.val * o.val
            result.err = ((self.err * o.val) ** 2 + (o.err * self.val) ** 2) ** 0.5
        else:
            try:
                c = float(o)
                result.val = self.val * c
                result.err = abs(c) * self.err
            except (ValueError, TypeError):
                return NotImplemented
        return result

    def __rmul__(self, o):
        return self.__mul__(o)

    def __truediv__(self, o):
        if o == 0:
            return Measure(float('nan'), float('nan'))
        result = Measure()
        if isinstance(o, Measure):
            result.val = self.val / o.val
            result.err = ((self.err / o.val) ** 2
                          + (o.err * self.val / o.val ** 2) ** 2) ** 0.5
        else:
            try:
                c = float(o)
                result.val = self.val / c
                result.err = self.err / abs(c)
            except (ValueError, TypeError):
                return NotImplemented
        return result

    def __rtruediv__(self, o):
        return (self ** -1).__mul__(o)

    def __pow__(self, o):
        result = Measure()
        if isinstance(o, Measure):
            result.val = self.val ** o.val
            result.err = abs(result.val) * (
                (o.val / self.val * self.err) ** 2
                + (math.log(self.val) * o.err) ** 2
            ) ** 0.5
        else:
            try:
                c = float(o)
                result.val = self.val ** c
                result.err = abs(result.val * c * self.err / self.val)
            except (ValueError, TypeError):
                return NotImplemented
        return result

    def __rpow__(self, o):
        try:
            c = float(o)
            val = c ** self.val
            err = abs(val * math.log(c) * self.err)
            return Measure(val, err)
        except (ValueError, TypeError):
            return NotImplemented

    def sqrt(self):
        """Return ``sqrt(self)`` with propagated uncertainty."""
        return self ** 0.5

    # ------------------------------------------------------------------
    # Comparison  (compares central values only)
    # ------------------------------------------------------------------

    def __abs__(self):
        return Measure(abs(self.val), self.err)

    def conjugate(self):
        return self

    def __eq__(self, o):
        if isinstance(o, Measure):
            return self.val == o.val
        try:
            return self.val == float(o)
        except (ValueError, TypeError):
            return NotImplemented

    def __ne__(self, o):
        eq = self.__eq__(o)
        return NotImplemented if eq is NotImplemented else not eq

    def __lt__(self, o):
        ref = o.val if isinstance(o, Measure) else float(o)
        return self.val < ref

    def __le__(self, o):
        ref = o.val if isinstance(o, Measure) else float(o)
        return self.val <= ref

    def __gt__(self, o):
        ref = o.val if isinstance(o, Measure) else float(o)
        return self.val > ref

    def __ge__(self, o):
        ref = o.val if isinstance(o, Measure) else float(o)
        return self.val >= ref

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __str__(self):
        dec = _err_dec(self.err) if self.err != 0 else _err_dec(self.val, 5)
        if _float_or_exp(self.val, dec) == 'e' and np.isfinite(self.err):
            try:
                precision = (1 + math.floor(math.log10(abs(self.val / self.err)))
                             if self.err != 0 else 2)
            except ValueError:
                precision = 2
            precision = max(precision, 2)
            val_s, err_s = _match_exponents(self.val, self.err, precision)
            e_str = f'{val_s} ± {err_s}'
            f_str = f'{self.val:.{dec}f} ± {self.err:.{dec}f}'
            if len(e_str) < len(f_str):
                return e_str
        return f'{self.val:.{dec}f} ± {self.err:.{dec}f}'

    def __repr__(self):
        return str(self)


# ------------------------------------------------------------------
# Module-level math helpers
# ------------------------------------------------------------------

def log(x, base=math.e):
    """
    Logarithm with error propagation.

    Parameters
    ----------
    x : Measure or float
    base : float

    Returns
    -------
    Measure or float
    """
    if isinstance(x, Measure):
        val = math.log(x.val, base)
        err = abs(x.err / (x.val * math.log(base)))
        return Measure(val, err)
    try:
        return math.log(float(x), base)
    except (ValueError, TypeError):
        return NotImplemented


# ------------------------------------------------------------------
# Private formatting helpers
# ------------------------------------------------------------------

def _err_dec(x, prec=2):
    """Number of decimal places needed to show ``prec`` significant figures of ``x``."""
    if math.isinf(x) or math.isnan(x) or x == 0:
        return 0
    dec = 0
    while int(abs(x)) < 10 ** (prec - 1):
        x *= 10
        dec += 1
    return dec


def _float_or_exp(x, dec, len_thresh=7):
    return 'e' if len(f'{x:.{dec}f}') > len_thresh else 'f'


def _get_exponent(value):
    if value == 0:
        return 0
    return int(math.floor(math.log10(abs(value))))


def _match_exponents(value1, value2, precision=2):
    exp = _get_exponent(value1)
    s1 = f'{value1 / 10 ** exp:.{precision}f}e{exp:+}'
    s2 = f'{value2 / 10 ** exp:.{precision}f}e{exp:+}'
    return s1, s2
