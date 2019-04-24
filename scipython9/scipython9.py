import math
import sys
import numpy as np
import copy
Polynomial = np.polynomial.Polynomial

#P9.1.1
def heron(a,b,c):
    s = 0.5 * (a+b+c)
    return math.sqrt(s * (s-a) * (s-b) * (s-c))

def kahan(a,b,c):
    el1 = a + (b+c)
    el2 = c - (a-b)
    el3 = c + (a-b)
    el4 = a + (b-c)
    return 0.25 * math.sqrt(el1*el2*el3*el4)
    
def comparison(a,b,c):
    print(heron(a,b,c), kahan(a,b,c))
    
#P9.1.2
def eps(p, implicit = True):
    #where p is the number of digits in the mantissa
    if implicit:
        p+1
    return 2**(-p)

#P9.2.1
def derivative():
    diff = np.zeros(3)
    def deriv(h):
        num = np.e**(1+h) - np.e
        return num / h
    for i in range(3):
        diff[i] = np.e - deriv(10**i)
    return np.argmin(diff)
    
#P6.2.2
def wilkinson():
    pass