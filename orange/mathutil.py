#	Mathematical utility routines
#	Copyright (C) 1999, Wesley Phoa
#
#	Reference: Numerical Recipes in C

# Although lacking the proper copyright notice, the author of this module
# states on his site (http://www.margaretmorgan.com/wesley/python/)
# that the code is released under GNU GPL

from operator import *
from Numeric import *

class BracketingException(Exception):
	pass

class RootFindingException(Exception):
	pass

class MinimizationException(Exception):
	pass

GOLDEN = (1+5**.5)/2
LITTLE = 1e-10
SQPREC = 1e-4

# 
# MISCELLANEOUS
#

def sgn(x):
	if x==0:
		return 0
	else:
		return x/abs(x)

def along(f, x, v):
	"""\
Given a multivariate function f, a point x and a vector v,
return the univariate function t |-> f(x+tv).
	"""
	return lambda t,f=f,x=x,v=v: apply(f, add(x, multiply(t, v)))

#
# UNIVARIATE ROOT FINDING
#

def bracket_root(f, interval, max_iterations=50):
	"""\
Given a univariate function f and a tuple interval=(x1,x2),
return a new tuple (bracket, fnvals) where bracket=(x1,x2)
brackets a root of f and fnvals=(f(x1),f(x2)).
	"""
	x1, x2 = interval
	if x1==x2:
		raise BracketingException("initial interval has zero width")
	elif x2<x1:
		x1, x2 = x2, x1
	f1, f2 = f(x1), f(x2)
	for j in range(max_iterations):
		while f1*f2 >= 0:  # not currently bracketed
			if abs(f1)<abs(f2):
				x1 = x1 + GOLDEN*(x1-x2)
			else:
				x2 = x2 + GOLDEN*(x2-x1)
			f1, f2 = f(x1), f(x2)
		return (x1, x2), (f1, f2)
	raise BracketingException("too many iterations")

def ridder_root(f, bracket, fnvals=None, accuracy=1e-6, max_iterations=50):
	"""\
Given a univariate function f and a tuple bracket=(x1,x2) bracketing a root,
find a root x of f using Ridder's method. Parameter fnvals=(f(x1),f(x2)) is optional.
	"""
	x1, x2 = bracket
	if fnvals==None:
		f1, f2 = f(x1), f(x2)
	else:
		f1, f2 = fnvals
	if f1==0:
		return x1
	elif f2==0:
		return x2
	elif f1*f2>=0:
		raise BracketingException("initial interval does not bracket a root")
	x4 = 123456789.
	for j in range(max_iterations):
		x3 = (x1+x2)/2
		f3 = f(x3)
		temp = f3*f3 - f1*f2
		x4, x4old = x3 + (x3-x1)*sgn(f1-f2)*f3/temp**.5, x4
		f4 = f(x4)
		if f1*f4<0:  # x1 and x4 bracket root
			x2, f2 = x4, f4
		else:  # x4 and x2 bracket root
			x1, f1 = x4, f4
		if min(abs(x1-x2),abs(x4-x4old))<accuracy or temp==0:
			return x4
	raise RootFindingException("too many iterations")

def root(f, interval=(0.,1.), accuracy=1e-4, max_iterations=50):
	"""\
Given a univariate function f and an optional interval (x1,x2),
find a root of f using bracket_root and ridder_root.
	"""
	bracket, fnvals = bracket_root(f, interval, max_iterations)
	return ridder_root(f, bracket, fnvals, accuracy, max_iterations)

#
# UNIVARIATE MINIMIZATION
#

def bracket_min(f, interval, max_iterations=50):
	"""\
Given a univariate function f and a tuple interval=(x1,x2),
return a new tuple (bracket, fnval) where bracket=(x1,x2,x3)
brackets a minimum of f and fnvals=(f(x1),f(x2),f(x3)).
	"""
	x1, x2 = interval
	f1, f2 = f(x1), f(x2)
	if f2>f1:  # ensure x1 --> x2 is downhill direction
		x1, x2, f1, f2 = x2, x1, f2, f1
	x3 = x2 + GOLDEN*(x2-x1)
	f3 = f(x3)
	for j in range(max_iterations):
		if f2<f3:
			if x1>x3:  # ensure x1<x2<x3
				x1, x3, f1, f3 = x3, x1, f3, f1
			return (x1, x2, x3), (f1, f2, f3)
		else:
			x1, x2, x3 = x1, x3, x3 + GOLDEN*(x3-x1)
			f1, f2, f3 = f(x1), f(x2), f(x3)
	raise BracketingException("too many iterations")

def brent_min(f, bracket, fnvals=None, tolerance=1e-6, max_iterations=50):
	"""\
Given a univariate function f and a tuple bracket=(x1,x2,x3) bracketing a minimum,
find a local minimum of f (with fn value) using Brent's method.
Optionally pass in the tuple fnvals=(f(x1),f(x2),f(x3)) as a parameter.
	"""
	x1, x2, x3 = bracket
	if fnvals==None:
		f1, f2, f3 = f(x1), f(xx), f(x3)
	else:
		f1, f2, f3 = fnvals
	if not f1>f2<f3:
		raise MinimizationException("initial triple does not bracket a minimum")
	if not x1<x3:  # ensure x1, x2, x3 in ascending order
		x1, f1, x3, f3 = x3, f3, x1, f1
	a, b = x1, x3

	e = 0.
	x = w = v = x2
	fw = fv = fx = f(x)

	for j in range(max_iterations):
		xm = (a+b)/2
		accuracy = tolerance*abs(x) + LITTLE
		if abs(x-xm) < (2*accuracy - (b-a)/2):
			return x, fx

		if abs(e)>accuracy:
			r = (x-w)*(fx-fv)
			q = (x-v)*(fx-fw)
			p = (x-v)*q - (x-w)*r
			q = 2*(q-r)
			if q>0:
				p = -p
			q = abs(q)
			etemp = e
			e = d
			if abs(p)>=abs(q*etemp)/2 or p<=q*(a-x) or p>=q*(b-x):
				if x>=xm:
					e = a-x
				else:
					e = b-x
				d = (2-GOLDEN)*e
			else:  # accept parabolic fit
				d = p/q
				u = x+d
				if u-a<2*accuracy or b-u<2*accuracy:
					d = accuracy*sgn(xm-x)
		else:
			if x>=xm:
				e = a-x
			else:
				e = b-x
			d = (2-GOLDEN)*e

		if abs(d)>=accuracy:
			u = x+d
		else:
			u = x+accuracy*sgn(d)
		fu = f(u)

		if fu<=fx:
			if u>=x:
				a = x
			else:
				b = x
			v, w, x = w, x, u
			fv, fw, fx = fw, fx, fu
		else:
			if u<x:
				a = u
			else:
				b = u
			if fu<-fw or w==x:
				v, w, fv, fw = w, u, fw, fu
			elif fu<=fw or v==x or v==w:
				v, fv = u, fu

	raise MinimizationException("too many iterations")

def minimum(f, interval=(0.,1.), tolerance=1e-4, max_iterations=50, return_fnval=0):
	"""\
Given a univariate function f and an optional interval (x1,x2),
find a local minimum of f using bracket_min and brent_min.
	"""
	bracket, fnvals = bracket_min(f, interval, max_iterations)
	min, fnval = brent_min(f, bracket, fnvals, tolerance, max_iterations)
	if return_fnval:
		return min, fnval
	else:
		return min

#
# MULTIVARIATE MINIMIZATION
#

def powell_min(f, p0, tolerance=1e-4, max_iterations=200, return_fnval=0):
	"""\
Given a multivariate function f and a starting point p0,
find a local minimum of f using Powell's direction set method.
	"""
	p = p0
	fp = apply(f, p)
	n = len(p)
	directions = identity(n).tolist()  # the n coordinate vectors

	for j in range(max_iterations):
		maxdrop_i = 0
		maxdrop = 0.
		pold=p
		fpold=fp

		for i in range(n):
			fptemp = fp
			v = directions[i]
			t, fp = minimum(along(f, p, v), return_fnval=1)
			p = add(p, multiply(t, v))
			if fptemp-fp>maxdrop:
				maxdrop_i = i
				maxdrop = fpold-fp

		totaldrop = fpold-fp
		if 2*totaldrop<=tolerance*(abs(fp)+abs(fpold)):
			if return_fnval:
				return tuple(p), fp
			else:
				return tuple(p)

		vnew = subtract(p, pold)
		pex = add(pold, multiply(2, vnew))
		fpex = apply(f, pex)
		if fpex<fp:
			temp = fpold-fp-maxdrop
			if 2*(fpold-2*fp+fpex)*temp*temp<totaldrop*totaldrop*maxdrop:
				t, fp = minimum(along(f, p, vnew), return_fnval=1)
				p = add(p, multiply(t, vnew))
				directions[maxdrop_i] = vnew

	raise MinimizationException("too many iterations")

if __name__=='__main__':

	from math import *

	print root(cos)
	print minimum(sin)

	def f(x, y):
		return x*x + x*y + y*y + 2*x - y

	print powell_min(f, (0., 0.))
