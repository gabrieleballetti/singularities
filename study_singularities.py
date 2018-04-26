from fractions import Fraction
import itertools
import numpy as np

# Create the space {0,1}^4 \ {(0,0,0,0)}
space = list(itertools.product(range(2), repeat=4))
del space[0]

# Create the list of monomials of degree j, for j in a range
multideg = [[m for m in itertools.product(range(5), repeat=4) if sum(m) == j] for j in range(5)]

class Monomial:
	def __init__(self, exponents, coefficient=1):
		self.dim = len(exponents)
		if not isinstance(coefficient, (int, Fraction, Polynomial)):
			raise TypeError("Coefficients should be in a ring (supported: int, Fraction, Polynomial)")
		else:
			self.c = coefficient
		if self.c == 0 or (isinstance(self.c, Polynomial) and self.c == Polynomial([])):
			self.c = 0
			self.e = [0 for i in range(self.dim)]
		else:
			self.e = [i for i in exponents]
	
	def __eq__(self, other):
		if type(self) != type(other):
			return False
		else:
			return self.e == other.e and self.c == other.c
	
	def __repr__(self, variables=[l for l in "xyzw"], coefficients=[l for l in "abcdefghijklmnopqrst"]):
		if all((exp == 0 for exp in self.e)):
			if isinstance(self.c, (int, Fraction)):
				return " " + str(self.c) + " "
			else:
				return " [" + self.c.__repr__(coefficients) + "]"
		if isinstance(self.c, (int, Fraction)):
			if self.c == 1:
				string = ""
			elif self.c == -1:
				string = " -"
			else:
				string = " " + str(self.c) + " *"
		else:
			string = " [" + self.c.__repr__(coefficients) + "] *"
		for i in zip(variables,self.e):
			if i[1] != 0:
				string += " " + i[0]
				if i[1] != 1:
					string += "**" + str(i[1])
		return string + " "
	
	def __lt__(self, other):
		if sum(self.e) != sum(other.e):
			return sum(self.e) < sum(other.e)
		else:
			for i, j in zip(self.e, other.e):
				if i != j:
					return i < j
		return False
	
	def __pos__(self):
		return self
	
	def __neg__(self):
		return Monomial(self.e, -self.c)
	
	def __mul__(self, other):
		if isinstance(other, (int, Fraction)):
			return Monomial(self.e,self.c * other)
		return Monomial([a + b for a, b in zip(self.e, other.e)] , self.c * other.c)
	
	def __rmul__(self, other):
		return self * other
	
	def evaluate(self, p):
		return self.c * int(np.prod([p[i]**(self.e[i]) for i in range(self.dim)]))
	
	def partial_derivative(self, i):
		new_exponents = [i for i in self.e]
		new_exponents[i] = max(new_exponents[i]-1,0)
		return Monomial(new_exponents, self.c * self.e[i])
	
	def partial_derivative_induced(self, d):
		new_monomial = Monomial(self.e,self.c)
		for i in range(new_monomial.dim):
			for j in range(d[i]):
				new_monomial = new_monomial.partial_derivative(i)
		return new_monomial
	
	def partial_derivatives(self, order=1):
		return [self.partial_derivative_induced(d) for d in multideg[order]]
	
	def multiplicity(self, p):
		order = 0
		while True:
			for dm in self.partial_derivatives(order):
				if dm.evaluate(p) != 0 and dm.evaluate(p) != Polynomial([]):
					return order
			order += 1
	
	def singularities(self):
		sing = {}
		for p in space:
			sing[p] = self.multiplicity(p)
		return sing


class Polynomial:
	def __init__(self, monomials):
		set_exps = {tuple(m.e) for m in monomials}
		list_mons = [Monomial(list(exp),sum([m.c for m in monomials if m.e == list(exp)])) for exp in set_exps]
		self.monomials = sorted([m for m in list_mons if m.c != 0], reverse=True)
		self.coefficients = [m.c for m in self.monomials]
	
	def __repr__(self, variables=[l for l in "xyzw"], coefficients=[l for l in "abcdefghijklmnopqrst"]):
		if len(self.monomials) == 0:
			return " 0 "
		string = "+".join((m.__repr__(variables, coefficients) for m in self.monomials))
		string = string.replace("+ -", "- ")
		string = string.replace("  ", " ")
		return string
	
	def __eq__(self, other):
		if type(self) != type(other):
			return False
		if len(self.monomials) != len(other.monomials):
			return False
		for m,n in zip(self.monomials,other.monomials):
			if tuple(m.e) != tuple(n.e):
				return False
		for c,d in zip(self.coefficients,other.coefficients):
			if c != d:
				return False
		return True
	
	def __ne__(self, other):
		return not self.__eq__(other)
	
	def __pos__(self):
		return self
	
	def __neg__(self):
		return Polynomial([Monomial(m.e,-m.c) for m in self.monomials])
	
	def __add__(self, other):
		if other == 0 or other == Polynomial([]):
			return self
		if isinstance(other, (int, Fraction)):
			return self + self.one() * other
		return Polynomial(self.monomials + other.monomials)
	
	def __radd__(self, other):
		return self + other
	
	def __sub__(self, other):
		return self + (-other)
	
	def __mul__(self, other):
		if isinstance(other, (int, Fraction)):
			return Polynomial([Monomial(m.e, other * m.c) for m in self.monomials])
		else:
			return sum([Polynomial([m*n for m in self.monomials]) for n in other.monomials])
	
	def __rmul__(self, other):
		return self * other
	
	def __pow__(self, exp):
		if self.monomials == []:
			return Polynomial([])
		p = self.one()
		for i in range(exp):
			p = p*self
		return p
	
	# Return the zero polynomial
	def zero(self):
		return Polynomial([])
	
	# Return the multiplicative identity of the ring of self
	def one(self):
		return Polynomial([Monomial([0 for i in self.monomials[0].e])])
	
	# Multiply by p the coefficients of self
	def mul_coeffs(self, other):
		return Polynomial([Monomial(m.e, m.c * other) for m in self.monomials])
	
	# Given a monomial (or a polynomial having only a monomial) p, returns them
	# coefficient of p in self. It ignores the coefficient of p.
	def coefficient(self, p):
		if isinstance(p, (int, Fraction)):
			p = p * self.one()
		elif isinstance(p, Monomial):
			p = Polynomial(p)
		if isinstance(p, Polynomial) and len(p.monomials) == 1:
			e = p.monomials[0].e
			return sum([m.c for m in self.monomials if m.e == e])
		
		def partial_derivative(self, i):
			return sum([Polynomial([m.partial_derivative(i)]) for m in self.monomials])
	
	# Set all the coefficients to 1
	def ignore_coeffs(self):
		if isinstance(self, (int, Fraction)):
			return self
		return Polynomial([Monomial(m.e) for m in self.monomials])
	
	def deg_le(self,k):
		return Polynomial([Monomial(m.e, m.c) for m in self.monomials if sum(m.e) <= k])
	
	def deg_eq(self,k):
		return Polynomial([Monomial(m.e, m.c) for m in self.monomials if sum(m.e) == k])
	
	def deg_ge(self,k):
		return Polynomial([Monomial(m.e, m.c) for m in self.monomials if sum(m.e) >= k])
	
	# Make self "general" by setting the coefficient of each monomial of
	# self to be a variable in a new ring
	def general(self):
		variable = lambda i, dim: Polynomial([Monomial([0 for j in range(i)] + [1] + [0 for j in range(i+1,dim)])])
		variables = [variable(i,len(self.monomials)) for i in range(len(self.monomials))]
		return Polynomial([Monomial(m.e,c) for m,c in zip(self.monomials,variables)])
	
	# Map the polynomial using the rule given.
	def change_coordinates(self, rule):
		if self.monomials == []:
			return Polynomial([])
		new_self = Polynomial([])
		for m in self.monomials:
			new_m = self.one()
			for i in range(len(m.e)):
				new_m = new_m * rule[i]**m.e[i]
			new_self += Polynomial([Monomial(n.e, m.c * n.c) for n in new_m.monomials])
		return new_self
	
	# Apply a change of coordinates to the coefficients of self, assuming that
	# they are polynomials
	def change_coordinates_coefficient(self, rule):
		return Polynomial([Monomial(m.e,m.c.change_coordinates(rule)) for m in self.monomials])
	
	# Return true if the support of self is bigger than the one of other
	def contains(self,other):
		set_self = {tuple(m.e) for m in self.monomials}
		set_other = {tuple(m.e) for m in other.monomials}
		return set_other <= set_self
	
	# Given a list of monomials f0, ... ,fh, return the s-fold base points
	# of the linear system S = l0*f0 + ... + lh*fh (with multiplicities)
	def base_points(self):
		points = {}
		for p in space:
			min_mult = min(m.multiplicity(p) for m in self.monomials)
			if min_mult > 1:
				points[p] = min_mult
		return points
	
	def local_equation(self, i):
		list_monomials = [Monomial(m.e[:i] + m.e[i+1:],m.c) for m in self.monomials]
		return Polynomial(list_monomials)


#------------------------------------------------------------------------------
# Useful functions
#------------------------------------------------------------------------------

def load_minimal_subpolytopes():
	with open('Dropbox/python/multiplicity/new_minimal_subpolytopes.txt', 'r') as f:
		list_msp = [eval(line) for line in f]
	return [Polynomial([Monomial(m) for m in p]) for p in list_msp]

list_minimal_polytopes = load_minimal_subpolytopes()

# Set some coefficients of f to be the equal to one()
def simplify_coefficients(f, monomials):
	new_monomials = []
	for m in f.monomials:
		if m.e in [p.monomials[0].e for p in monomials]:
			new_monomials.append(Monomial(m.e,m.c.one()))
		else:
			new_monomials.append(m)
	return Polynomial(new_monomials)

#------------------------------------------------------------------------------
# Fix some 'famous' polynomial
#------------------------------------------------------------------------------

x = Polynomial([Monomial([1,0,0])])
y = Polynomial([Monomial([0,1,0])])
z = Polynomial([Monomial([0,0,1])])
zero = Polynomial([])


#------------------------------------------------------------------------------
# Check the 2-jet
#------------------------------------------------------------------------------

# Apply a permutation of the indeterminates in order to put the
# 2-jet in one of the "nice" forms 
def make_it_beautiful(f):
	list_beautiful_2jets = [
		x * z + y**2 + y * z + z**2,
		x * y + x * z + y * z,
		x**2 + y * z,
		x**2 + x * z + y * z,
		x * y + x * z + y * z + z**2,
		x * y,
		x * z + y * z,
		y**2 + y * z + z**2,
		x * z + y * z + z**2,
		x**2 + x * y,
		x**2
	]
	possible_perms = [list(itertools.permutations(m.e)) for m in f.monomials]
	for i in range(len(possible_perms[0])):
		new_jet_mons = [list(m[i]) for m in possible_perms]
		new_f = Polynomial([Monomial(e,c) for e,c in zip(new_jet_mons,f.coefficients)])
		new_jet = new_f.deg_le(2)
		for fi in list_beautiful_2jets:
			if new_jet == fi:
				return new_f

#------------------------------------------------------------------------------
# Rank 1
#------------------------------------------------------------------------------
# list of possible 2-jets of rank 1
rank_1 = [
	x**2
]

# Given a general polynomial f(x,y,z) = x^2 + f_(>=3)(y,z) + x g(x,y,z),
# transforms it f via the map x -> x - 1/2 * g(x,y,z)
def transform(f):
	f3 = Polynomial([Monomial(m.e, m.c) for m in f.deg_ge(3).monomials if m.e[0] == 0])
	g = Polynomial([Monomial([m.e[0]-1,m.e[1],m.e[2]], m.c) for m in f.monomials if m.e[0] >= 1 and m.e != [2,0,0]])
	x2 = f.deg_le(2)
	assert f == x2 + f3 + x*g, "ERROR"
	f = f.change_coordinates([x - Fraction(1,2) * g, y, z])
	return f

# f(3) having exactly three distinct linear factors
f3_3_linear_factors = [
	y**2 * z + y * z**2,
	y**3 + y**2 * z + y * z**2,
	y**2 * z + y * z**2 + z**3
]
# possible f(3) having exactly two distinct linear factors
f3_2_linear_factors = [
	 y * z**2 + z**3,
	 y**3 + y**2 * z,
	 y * z**2,
	 y**2 * z
]
# possible f(3) having one linear factors
f3_1_linear_factors = [
	 y**3,
	 z**3
]

def study_singularity_rank_1(f):
	print('We simplify the coefficient of x**2 (using x -> 1/a * x)')
	f = simplify_coefficients(f, [x**2])
	print(f)
	print('We now apply the transformation x -> x - 1/2 * g(x,y,z)')
	f = transform(f)
	print('[...]')
	if f.ignore_coeffs().deg_eq(3) in f3_3_linear_factors:
		return 'D_4'
	elif f.ignore_coeffs().deg_eq(3) in f3_2_linear_factors:
		# apply a change of coordinates to put the degree 3 part
		# in the form yz^2 + z^3
		if f.contains(y * z**2):
			f = f.change_coordinates([x,z,y])
		# apply a change of coordinates to remove the monomial
		# y^3, if present
		if f.contains(y**3):
			f = simplify_coefficients(f, [y**2 * z, y**3])
			f = f.change_coordinates([x,y,z-y])
		# Now the 3-jet is of the form x^2 + y^2 z. We now just
		# consider the part of degree >= 3 of f in y and z
		if f.contains(z**4):
			return 'D_5'
		elif f.contains(y * z**3):
			return 'D_6'
		elif f.contains(y * z**4) or f.contains(z**6):
			return 'D_7'
		else:
			assert False, 'Cannot find the Dk singularity'
	elif f.ignore_coeffs().deg_eq(3) in f3_1_linear_factors:
		# Now f has the form x^2 + y^3 + ... 
		if f.ignore_coeffs().deg_eq(3) == z**3:
			f = f.change_coordinates([x,z,y])
		if f.contains(y**2 * z) or f.contains(y * z**2) or f.contains(y * z**3):
			if f.contains(z**4):
				return 'E_6'
			else:
				if f.contains(y * z**3):
					return 'E_7'
				else:
					return 'E_8'
		else:
			assert False, 'Cannot find the Ek singularity'
	else:
		assert False, 'Something wrong with the linear factors'

#------------------------------------------------------------------------------
# Rank 2
#------------------------------------------------------------------------------
# list of possible 2-jets of rank 2
rank_2 = [
	x * y,
	x * z + y * z,
	y**2 + y * z + z**2,
	x * z + y * z + z**2,
	x**2 + x * y
]

# linear transformations transforming each of the 2-jets above into x**2 + y**2
# x * y
def r2_0(f):
	f = simplify_coefficients(f, [x * y])
	f = f.change_coordinates([x + y, x - y, z])
	return simplify_coefficients(f, [y**2])

# x * z + y * z
def r2_1(f):
	f = simplify_coefficients(f, [x * z, y * z])
	f = f.change_coordinates([x + z,x - z, y])
	return r2_0(f)

# y**2 + y * z + z**2
def r2_2(f):
	f = simplify_coefficients(f, [y**2, y * z])
	f = f.change_coordinates([z, y + x, -2 * x])
	return simplify_coefficients(f, [x**2])

# x * z + y * z + z**2
def r2_3(f):
	f = simplify_coefficients(f, [x * z, y * z, z**2])
	f = f.change_coordinates([x, -x + z, y]).change_coordinates([z, x + y, -2 * y])
	return simplify_coefficients(f, [y**2])

# x**2 + x * y
def r2_4(f):
	f = simplify_coefficients(f, [x**2, x * y])
	f = f.change_coordinates([x + y, -2 * y, z])
	return simplify_coefficients(f, [y**2])

linear_changes = [r2_0, r2_1, r2_2, r2_3, r2_4]

def study_singularity_rank_2(f):
	type_2jet = f.deg_le(2).ignore_coeffs()
	# Apply a change of coordinates to bring the 2-jet in the
	# form x^2 + y^2
	print('f has this kind of 2-jet' + type_2jet.__repr__())
	linear_change = linear_changes[rank_2.index(type_2jet)]
	# In the linear changes we also simplify some coefficients
	# using trasformations like x -> k * x
	f = linear_change(f)
	print('we apply a linear transformation to make it into x**2 + y**2, in the process we might have simplified some coefficient via transformations x -> k * x')
	print(f)
	assert f.deg_le(2).ignore_coeffs() == x**2 + y**2, "Error in the trasformation of the 2-jet"
	print('Now we find the Ak singularity by checking if some monomials are contained in f')
	if f.contains(z**3):
		return 'A_2'
	elif f.contains(z**4) or f.contains(x * z**2) or f.contains(y * z**2):
		return 'A_3'
	elif f.contains(x * z**3) or f.contains(y * z**3):
		return 'A_5'
	else:
		assert False, 'Cannot find the Ak singularity'

#------------------------------------------------------------------------------
# Rank 3
#------------------------------------------------------------------------------
rank_3 = [
	x * z + y**2 + y * z + z**2,
	x * y + x * z + y * z,
	x**2 + y * z,
	x**2 + x * z + y * z,
	x * y + x * z + y * z + z**2
]

def study_singularity_rank_3(f):
	return 'A_1'

#------------------------------------------------------------------------------
# Study the singularity
#------------------------------------------------------------------------------
def study_singularity(f,p):
	print(p)
	print('Local equation wrt the point above:' + f.local_equation(p.index(1)).__repr__())
	f = make_it_beautiful(f.local_equation(p.index(1)))
	print('We permute the variables to make the 2-jet beautiful')
	print(f)
	f = f.general()
	print('We make f general by introducing a variable for each monomial')
	print(f)
	type_2jet = f.deg_le(2).ignore_coeffs()
	if type_2jet in rank_3:
		print("the 2-jet has rank 3")
		return study_singularity_rank_3(f)
	elif type_2jet in rank_2:
		print("the 2-jet has rank 2")
		return study_singularity_rank_2(f)
	elif type_2jet in rank_1:
		print("the 2-jet has rank 1")
		return study_singularity_rank_1(f)
	else:
		assert False, 'The 2-jet is not beautiful'

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

list_sing = []
for f in list_minimal_polytopes:
	print('----------------------------------------------------------------------------------------------------')
	print('Starting polynomial: ' + f.__repr__())
	print('Base points (with multiplicity):')
	print(f.base_points())
	for p in f.base_points():
		sing = study_singularity(f,p)
		sing
		assert sing in ['A_1','A_2','A_3','A_5','D_4','D_5','D_6','D_7','E_6','E_7','E_8']
		list_sing.append(sing)
		print(' ')

from collections import Counter
Counter(list_sing)

#Counter({'A_3': 127, 'A_5': 58, 'A_2': 32, 'D_5': 26, 'A_1': 22, 'E_6': 22, 'D_4': 14, 'D_6': 12, 'D_7': 10, 'E_7': 9})






















