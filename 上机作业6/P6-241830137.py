import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_legendre
from numpy.polynomial.legendre import leggauss

def f(x):
    return np.exp(x)

def power_basis_coeffs(n, high_precision=False):
    if high_precision:
        from decimal import Decimal, getcontext
        getcontext().prec = 50
        e = Decimal(np.e).exp()
        b = [Decimal(0)] * (n + 1)
        b[0] = e - Decimal(1)
        for j in range(1, n + 1):
            b[j] = e - Decimal(j) * b[j-1]
        H = np.zeros((n + 1, n + 1), dtype=object)
        for i in range(n + 1):
            for j in range(n + 1):
                H[i, j] = Decimal(1) / Decimal(i + j + 1)
        a = np.linalg.solve(H.astype(float), np.array(b, dtype=float))
        return a
    else:
        b = np.zeros(n + 1)
        b[0] = np.e - 1
        for j in range(1, n + 1):
            b[j] = np.e - j * b[j-1]
        H = np.fromfunction(lambda i, j: 1.0 / (i + j + 1), (n + 1, n + 1))
        a = np.linalg.solve(H, b)
        return a

def power_approx(x, coeffs):
    return np.polyval(coeffs[::-1], x)

def legendre_coeffs(n, method='gauss', N_gauss=50):
    e_half = np.exp(0.5)
    if method == 'gauss':
        t, w = leggauss(N_gauss)
        ft = e_half * np.exp(t / 2.0)
        coeffs = np.zeros(n + 1)
        for k in range(n + 1):
            Pk = eval_legendre(k, t)
            integral = np.sum(w * ft * Pk)
            coeffs[k] = (2 * k + 1) / 2.0 * integral
        return coeffs
    elif method == 'recur':
        e_n = np.exp(-0.5)
        I = np.zeros(n + 1)
        I[0] = 2.0 * (e_half - e_n)
        if n >= 1:
            I[1] = 6.0 * e_half + 2.0 * e_n
        for k in range(1, n):
            term = e_half - ((-1) ** k) * e_n
            I[k+1] = 2.0 * (2 * k + 1) * I[k] - I[k-1] - 2.0 * (2 * k + 1) * term
        coeffs = (2.0 * np.arange(n + 1) + 1) / 2.0 * e_half * I
        return coeffs

def legendre_approx(x, coeffs):
    t = 2.0 * x - 1.0
    n = len(coeffs) - 1
    val = np.zeros_like(x)
    for k in range(n + 1):
        val += coeffs[k] * eval_legendre(k, t)
    return val

def compute_errors(x_test, y_true, y_approx):
    abs_err = np.abs(y_true - y_approx)
    max_err = np.max(abs_err)
    l2_err = np.sqrt(np.trapezoid((y_true - y_approx)**2, x_test))
    return max_err, l2_err

def plot_results(x_plot, y_true, approx_dict, title):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x_plot, y_true, 'k-', linewidth=2, label='$e^x$')
    for label, y_app in approx_dict.items():
        plt.plot(x_plot, y_app, '--', label=label)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title + ' - approximation')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    for label, y_app in approx_dict.items():
        err = np.abs(y_true - y_app)
        plt.semilogy(x_plot, err, label=label)
    plt.xlabel('x')
    plt.ylabel('absolute error')
    plt.title(title + ' - error (log)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'figure_n{title.split("=")[1].strip()}.pdf', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    n_list = [5, 10]
    x_plot = np.linspace(0, 1, 500)
    y_true = f(x_plot)
    for n in n_list:
        print(f"\n{'='*60}\n n = {n}\n{'='*60}")
        a_pow = power_basis_coeffs(n, high_precision=False)
        print(f"\npower coeffs (n={n}):")
        for k, ak in enumerate(a_pow):
            print(f"  a_{k} = {ak:15.10f}")
        y_pow = power_approx(x_plot, a_pow)
        max_err_pow, l2_err_pow = compute_errors(x_plot, y_true, y_pow)
        print(f"max error: {max_err_pow:.4e}, L2 error: {l2_err_pow:.4e}")
        if n == 10:
            a_pow_hp = power_basis_coeffs(n, high_precision=True)
            print("\npower coeffs (high precision, n=10):")
            for k, ak in enumerate(a_pow_hp):
                print(f"  a_{k} = {ak:15.10f}")
        c_leg_gauss = legendre_coeffs(n, method='gauss')
        print(f"\nLegendre coeffs (Gauss, n={n}):")
        for k, ck in enumerate(c_leg_gauss):
            print(f"  c_{k} = {ck:15.10f}")
        y_leg_gauss = legendre_approx(x_plot, c_leg_gauss)
        max_err_leg, l2_err_leg = compute_errors(x_plot, y_true, y_leg_gauss)
        print(f"max error: {max_err_leg:.4e}, L2 error: {l2_err_leg:.4e}")
        c_leg_recur = legendre_coeffs(n, method='recur')
        print(f"\nLegendre coeffs (recur, n={n}):")
        for k, ck in enumerate(c_leg_recur):
            print(f"  c_{k} = {ck:15.10f}")
        y_leg_recur = legendre_approx(x_plot, c_leg_recur)
        max_err_leg2, l2_err_leg2 = compute_errors(x_plot, y_true, y_leg_recur)
        print(f"max error: {max_err_leg2:.4e}, L2 error: {l2_err_leg2:.4e}")
        approx_dict = {
            f'power (n={n})': y_pow,
            f'Legendre Gauss (n={n})': y_leg_gauss,
        }
        plot_results(x_plot, y_true, approx_dict, f'n = {n}')
    print("\n" + "="*60)
    print("Hilbert matrix condition numbers (power basis):")
    for n in [5, 10]:
        H = np.fromfunction(lambda i, j: 1.0/(i+j+1), (n+1, n+1))
        print(f"n={n}: cond(H) = {np.linalg.cond(H):.2e}")
    print("Legendre basis condition number = 1")