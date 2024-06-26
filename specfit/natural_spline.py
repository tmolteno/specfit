import sympy as sp
import matplotlib.pyplot as plt 
import numpy as np
import scipy

## Cubic spline 

## Changepoint pymc https://ckrapu.github.io/blog/2022/nonparametric-changepoint-model-pymc/
### https://causalpy.readthedocs.io/en/latest/notebooks/rkink_pymc.html

def poly(x, x_0, x_1, n, degree=3):
    #
    # Create a polynomial that will be bounded between x_0, and x_1 which is the n_th spline interval.
    #
    a = [sp.Symbol(f"a_{n}_{i}") for i in range(degree+1)]

    _poly = 0
    for i in range(degree+1):
        _poly += a[i]*x**i
    return _poly, a


def window_below(x, x_1):
    return (1 - sp.Heaviside(x - x_1))


def window_above(x, x_0):
    return sp.Heaviside(x - x_0)


def window(x, x_0, x_1):
    return sp.Heaviside(x - x_0) * (1 - sp.Heaviside(x - x_1))


def natural_spline(x, knot_x, knot_y, degree):

    # The sum of piecewise step functions each that contains a cubic polynomial. We don't include the boundary knots

    n_knots = len(knot_x)
    n_intervals = n_knots - 1
    polys = []
    variables = []
    windows = []
    for i in range(n_intervals):
        x_0 = knot_x[i]
        x_1 = knot_x[i+1]
        print(f"interval {i} : {x_0},{x_1}")
        p, pvars = poly(x, x_0, x_1, i, degree)
        polys.append(p)
        variables.extend(pvars)
        if i == 0:
            windows.append(window_below(x, x_1))
        elif i == n_intervals-1:
            windows.append(window_above(x, x_0))
        else:
            windows.append(window(x, x_0, x_1))

    eqns = []
    for i in range(1, n_knots-1):
        x_0 = knot_x[i]
        y_0 = knot_y[i]
        print(f"interior knots {i} : {x_0}")
        p_0 = polys[i-1]
        p_1 = polys[i]

        # Set interior values
        eqns.append(sp.Eq(p_0.subs(x, x_0), y_0))
        eqns.append(sp.Eq(p_1.subs(x, x_0), y_0))
        # Match interior points
        eqns.append(sp.Eq(p_0.subs(x, x_0), p_1.subs(x, x_0)))
        # Match interior derivatives
        for d in range(degree):
            eqns.append(sp.Eq(sp.diff(p_0, x, d).subs(x, x_0),
                              sp.diff(p_1, x, d).subs(x, x_0)))

    for i in [0, -1]:  # Boundary knots
        # Second derivatives to zero at the boundary
        eqns.append(sp.Eq(sp.diff(polys[i], x, 2).subs(x, knot_x[i]), 0))
        # Set exterior points to their values
        eqns.append(sp.Eq(polys[i].subs(x, knot_x[i]), knot_y[i]))

    soln = sp.solve(eqns, variables)
    reduced_polys = []
    for p in polys:
        rp = sp.simplify(p.subs(soln))
        reduced_polys.append(rp)

    spline = 0
    for p, w in zip(reduced_polys, windows):
        spline = spline + w * p

    expr = sp.simplify(spline.subs(soln))
    free_syms = sorted(expr.free_symbols, key = lambda symbol: symbol.name)

    return expr, eqns, free_syms, reduced_polys


class NaturalCubicSpline:
    #
    #   A natural cubic spline that can be evaluated using numpy, with unknown knot positions
    #
    def __init__(self, interval, n_interior_knots, degree=3):

        x = sp.Symbol('x')

        x_0, x_1 = interval

        k = [sp.Symbol(f"k_{n}") for n in range(n_interior_knots)]
        all_knots = [[x_0], k, [x_1]]
        print(all_knots)

        knot_x = [item for row in all_knots for item in row]
        knot_y = [sp.Symbol(f"y_{n}") for n in range(len(knot_x))]

        self.expr, self.eqns, self.variables, self.polys = \
            natural_spline(x, knot_x=knot_x,
                           knot_y=knot_y,
                           degree=degree)
        print(self.variables)
        print(self.eqns)
        print(self.expr)
        for p in self.polys:
            print(p)
        self.n_interior_knots = n_interior_knots
        self.x_0, self.x_1 = interval
        self.degree = degree
        self.n_param = len(self.variables) - 1

    def get_expr(self):
        f = sp.lambdify(self.variables, self.expr, cse=True)
        return f

    def regression(self, x_data, y_data):
        # Calculate the best fit spline to match the data in a least squares sense
        expr = self.get_expr()
        # The arguments of expr, are in self.variables (in order), these are {k_i}, x, {y_i}
        # The parameters are {k_i}{y_i}

        def f(v):
            param = v[0:self.n_interior_knots].tolist()
            param.append(x_data)
            param.extend(v[self.n_interior_knots:].tolist())
            residual = expr(*param) - y_data
            return np.sum(residual**2)


        bounds = []
        v0 = []
        for i in range(1, self.n_interior_knots+1):
            bounds.append((self.x_0, self.x_1))
            v0.append(self.x_0 + i*(self.x_1 - self.x_0)/(self.n_interior_knots+1))

        n_knots = self.n_param - self.n_interior_knots
        for i in range(n_knots):
            bounds.append((None, None))
            ind = int(i*y_data.shape[0]/n_knots)
            print(ind)
            v0.append(y_data[ind])
        print(f"v0 = {v0}, var={self.variables}")

        v0 = np.array(v0)
        f0 = f(v0)
        print(f0)

        opt = scipy.optimize.minimize(f, v0, method="Nelder-Mead",
                                      bounds=bounds)
        print(opt)
        v_best = opt.x

        ret = {}
        ret["interior_knots"] = v_best[0:self.n_interior_knots].tolist()
        ret["interval"] = [self.x_0, self.x_1]
        ret["degree"] = self.degree

        k_x = []
        k_x.append(self.x_0)
        k_x.extend(v_best[0:self.n_interior_knots].tolist())
        k_x.append(self.x_1)

        ret["knot_x"] = k_x
        ret["knot_y"] = v_best[self.n_interior_knots:].tolist()

        return ret


if __name__ == "__main__":
    spline = NaturalCubicSpline([-1, 3], n_interior_knots=1, degree=3)

    f = spline.get_expr()

    x = np.linspace(-1,3,20)
    y = np.sin(x)

    ret = spline.regression(x, y)
    print(ret)

    plt.plot(x, y, '.', label="data")

    knot_y = ret["knot_y"]
    y_best = f(k_0=ret["interior_knots"][0], x=x, y_0=knot_y[0],
               y_1=knot_y[1],
               y_2=knot_y[2])

    plt.plot(x, y_best, label="spline regression")
    plt.legend()
    # plt.plot(2.5, 2, 'o')
    plt.show()
