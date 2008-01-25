all: tmatrix

# tmatrix: lpd.f ampld.par.f ampld.lp.f modified_double_precision_drop.f ampld.lp.pyf
# 	f2py -c -m tmatrix ampld.lp.pyf ampld.lp.f modified_double_precision_drop.f \
#   lpd.f

tmatrix: ampld.par.f ampld.lp.f modified_double_precision_drop.f ampld.lp.pyf
	f2py -c -m _tmatrix ampld.lp.pyf ampld.lp.f modified_double_precision_drop.f \
	-llapack -lcblas

install: all
	cp _tmatrix.so ..

