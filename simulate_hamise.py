# simulate_hamise.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta, norm
import statsmodels.api as sm
from typing import Tuple, List

np.random.seed(123)

def m_true(x: np.ndarray) -> np.ndarray:
    # regression function: m(x) = sin( (x/3 + 0.1)^{-1} )
    return np.sin((x / 3.0 + 0.1) ** -1.0)

def generate_data(n: int, alpha: float, beta_param: float, sigma: float) -> Tuple[np.ndarray,np.ndarray]:
    X = beta.rvs(alpha, beta_param, size=n)
    Y = m_true(X) + norm.rvs(scale=sigma, size=n)
    return X, Y

def block_indices_sorted(X: np.ndarray, N: int) -> List[np.ndarray]:
    # sort by X and split into N nearly equal blocks
    n = len(X)
    order = np.argsort(X)
    sizes = [n // N + (1 if i < (n % N) else 0) for i in range(N)]
    inds = []
    pos = 0
    for s in sizes:
        inds.append(order[pos:pos+s])
        pos += s
    return inds

def fit_quartic_block(X_block: np.ndarray, Y_block: np.ndarray):
    # design matrix for 1, x, x^2, x^3, x^4
    Xmat = np.vstack([np.ones_like(X_block), X_block, X_block**2, X_block**3, X_block**4]).T
    model = sm.OLS(Y_block, Xmat).fit()
    return model.params, model

def compute_m2_at_x(beta_coef: np.ndarray, x: np.ndarray) -> np.ndarray:
    # second derivative of b0 + b1 x + b2 x^2 + b3 x^3 + b4 x^4
    # m''(x) = 2*b2 + 6*b3*x + 12*b4*x^2
    b2, b3, b4 = beta_coef[2], beta_coef[3], beta_coef[4]
    return 2.0*b2 + 6.0*b3*x + 12.0*b4*(x**2)

def estimate_theta22_sigma2(X: np.ndarray, Y: np.ndarray, N: int, verbose=False):
    n = len(X)
    inds_blocks = block_indices_sorted(X, N)
    m2_vals_squared = np.zeros(n)
    residuals_sq = np.zeros(n)
    for j,inds in enumerate(inds_blocks):
        Xb = X[inds]
        Yb = Y[inds]
        # To avoid singular problems, if block too small fallback to polyfit with regularization.
        if len(inds) < 5:
            # use numpy polyfit degree 4 (it works with small blocks too)
            coefs = np.polyfit(Xb, Yb, 4)
            # np.polyfit returns coeffs for x^4,..., const
            # convert to b0..b4
            b4, b3, b2, b1, b0 = coefs
            beta_coef = np.array([b0,b1,b2,b3,b4])
            # build fitted values and residuals
            Yhat = np.polyval(coefs, Xb)
        else:
            beta_coef, model = fit_quartic_block(Xb, Yb)
            Yhat = model.fittedvalues
        
        m2 = compute_m2_at_x(beta_coef, Xb)
        m2_vals_squared[inds] = m2**2
        residuals_sq[inds] = (Yb - Yhat)**2
    
    theta22_hat = m2_vals_squared.mean()  # 1/n sum (m''(Xi))^2
    sigma2_hat = residuals_sq.sum() / (n - 5*N)
    RSS = residuals_sq.sum()
    if verbose:
        print(f"N={N}, theta22_hat={theta22_hat:.6g}, sigma2_hat={sigma2_hat:.6g}, RSS={RSS:.6g}")
    return theta22_hat, sigma2_hat, RSS

def h_AMISE_hat(n: int, sigma2_hat: float, theta22_hat: float, support_len: float=1.0):
    # plug-in formula (quartic kernel) given in assignment:
    # h = n^{-1/5} * (35 * sigma2 * |supp(X)| / theta22)^{1/5}
    return n**(-1/5) * ((35.0 * sigma2_hat * support_len / theta22_hat) ** (1.0/5.0))

def N_max_rule(n: int) -> int:
    return max(min(n//20, 5), 1)

def choose_N_via_Cp(X: np.ndarray, Y: np.ndarray, N_candidates: List[int]):
    n = len(X)
    # compute RSS for all N
    records = []
    RSS_dict = {}
    for N in N_candidates:
        _, _, RSS = estimate_theta22_sigma2(X, Y, N)
        RSS_dict[N] = RSS
    Nmax = max(N_candidates)
    # denominator in Cp uses RSS(N_max)/(n - 5*N_max)
    denom = RSS_dict[Nmax] / (n - 5*Nmax)
    Cp_vals = {}
    for N in N_candidates:
        Cp = RSS_dict[N] / denom - (n - 10*N)
        Cp_vals[N] = Cp
    best_N = min(Cp_vals, key=Cp_vals.get)
    return best_N, Cp_vals, RSS_dict

# Example single-run demonstration
def single_run_demo(n=500, alpha=2.0, beta_param=2.0, sigma=1.0, plot=True):
    X, Y = generate_data(n, alpha, beta_param, sigma)
    # candidate N's (1 .. N_max)
    Nmax = N_max_rule(n)
    N_candidates = list(range(1, Nmax+1))
    bestN, Cp_vals, RSS_dict = choose_N_via_Cp(X, Y, N_candidates)
    print("N candidates:", N_candidates, "bestN by Cp:", bestN)
    theta_hat, sigma2_hat, _ = estimate_theta22_sigma2(X, Y, bestN, verbose=True)
    support_len = X.max() - X.min()  # empirical support
    h_hat = h_AMISE_hat(n, sigma2_hat, theta_hat, support_len=support_len)
    print(f"Estimated h_AMISE: {h_hat:.6g}")
    if plot:
        # plot m_true, data, and block-fits for chosen N
        sns.set(style='whitegrid')
        xs = np.linspace(0,1,400)
        plt.figure(figsize=(8,5))
        plt.scatter(X, Y, s=10, alpha=0.6, label='data')
        plt.plot(xs, m_true(xs), 'k-', lw=2, label='m_true')
        # plot block polynomial fits
        inds_blocks = block_indices_sorted(X, bestN)
        for inds in inds_blocks:
            Xb = X[inds]
            Yb = Y[inds]
            try:
                beta_coef, model = fit_quartic_block(Xb, Yb)
                # build poly for plot
                coefs = np.array([beta_coef[4], beta_coef[3], beta_coef[2], beta_coef[1], beta_coef[0]])
                xs_block = np.linspace(Xb.min(), Xb.max(), 50)
                ys_block = np.polyval(coefs, xs_block)
                plt.plot(xs_block, ys_block, lw=2)
            except Exception as e:
                continue
        plt.title(f"n={n}, alpha={alpha}, beta={beta_param}, sigma={sigma}, N={bestN}, h_hat={h_hat:.3f}")
        plt.legend()
        plt.xlabel('X'); plt.ylabel('Y')
        plt.show()
    return {'n': n, 'alpha':alpha, 'beta':beta_param, 'sigma':sigma, 'bestN':bestN, 'h_hat':h_hat,
            'theta_hat':theta_hat, 'sigma2_hat':sigma2_hat, 'Cp_vals':Cp_vals, 'RSS':RSS_dict}

# Monte Carlo experiment function
def run_experiment_grid(n_list, alpha_beta_list, sigma=1.0, repetitions=50):
    results = []
    for n in n_list:
        for (alpha,beta_param) in alpha_beta_list:
            for rep in range(repetitions):
                X, Y = generate_data(n, alpha, beta_param, sigma)
                # choose Nset from 1..Nmax
                Nmax = N_max_rule(n)
                N_candidates = list(range(1, Nmax+1))
                bestN, Cp_vals, RSS_dict = choose_N_via_Cp(X, Y, N_candidates)
                theta_hat, sigma2_hat, _ = estimate_theta22_sigma2(X, Y, bestN)
                support_len = X.max() - X.min()
                h_hat = h_AMISE_hat(n, sigma2_hat, theta_hat, support_len=support_len)
                results.append({'n':n, 'alpha':alpha, 'beta':beta_param, 'rep':rep,
                                'bestN':bestN, 'h_hat':h_hat,
                                'theta_hat':theta_hat, 'sigma2_hat':sigma2_hat})
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Quick demo
    out = single_run_demo(n=400, alpha=2.0, beta_param=2.0, sigma=1.0, plot=True)
    print(out)
    # A small experiment grid (for demonstration; increase 'repetitions' for real study)
    df = run_experiment_grid(n_list=[200, 500], alpha_beta_list=[(0.5,0.5),(2,2),(5,1)], sigma=1.0, repetitions=20)
    # basic summaries
    print(df.groupby(['n','alpha','beta'])['h_hat'].agg(['mean','std','count']))
    # quick boxplot
    plt.figure(figsize=(10,5))
    sns.boxplot(data=df, x='n', y='h_hat', hue='alpha')
    plt.title('Distribution of estimated h_hat by n and alpha (demo)')
    plt.show()

# Exemple : effect of N
n = 500
alpha, beta_param, sigma = 2, 2, 1.0
X, Y = generate_data(n, alpha, beta_param, sigma)

N_candidates = list(range(1, N_max_rule(n)+1))
theta_list, sigma_list, h_list, Cp_list = [], [], [], []

_, _, RSS_ref = estimate_theta22_sigma2(X, Y, N_candidates[-1])
denom = RSS_ref / (n - 5*N_candidates[-1])

for N in N_candidates:
    theta_hat, sigma2_hat, RSS = estimate_theta22_sigma2(X, Y, N)
    h_hat = h_AMISE_hat(n, sigma2_hat, theta_hat)
    Cp = RSS / denom - (n - 10*N)
    theta_list.append(theta_hat)
    sigma_list.append(sigma2_hat)
    h_list.append(h_hat)
    Cp_list.append(Cp)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(N_candidates, h_list, marker="o")
plt.xlabel("N")
plt.ylabel(r"$\hat h$")
plt.title("Estimated bandwidth vs N")

plt.subplot(1,2,2)
plt.plot(N_candidates, Cp_list, marker="o", color="red")
plt.xlabel("N")
plt.ylabel(r"$C_p(N)$")
plt.title("Mallows Cp vs N")
plt.tight_layout()
plt.show()

# Size of n effect
df_n = run_experiment_grid(
    n_list=[100,200,500,1000],
    alpha_beta_list=[(2,2)],  # distribution symétrique
    sigma=1.0,
    repetitions=50
)

plt.figure(figsize=(8,5))
sns.boxplot(data=df_n, x="n", y="h_hat")
plt.title("Effect of sample size on bandwidth estimates")
plt.ylabel(r"$\hat h$")
plt.show()

# Effet de la distribution X
df_beta = run_experiment_grid(
    n_list=[500],
    alpha_beta_list=[(0.5,0.5),(2,2),(5,1)],
    sigma=1.0,
    repetitions=50
)

plt.figure(figsize=(8,5))
sns.boxplot(data=df_beta, x="alpha", y="h_hat", hue="beta")
plt.title("Effect of Beta distribution of X on bandwidth estimates")
plt.ylabel(r"$\hat h$")
plt.show()

xs = np.linspace(0,1,200)
plt.figure(figsize=(8,5))
for (a,b) in [(0.5,0.5),(2,2),(5,1)]:
    plt.plot(xs, beta.pdf(xs,a,b), label=f"Beta({a},{b})")
plt.title("Different Beta densities for X")
plt.legend()
plt.show()
