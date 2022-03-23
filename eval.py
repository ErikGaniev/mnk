from lsp import lstsq
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

A = np.random.uniform(low=-100, high=100, size=(500, 20))
x = np.random.uniform(low=-100, high=100, size=(20, 1))
prod = A @ x
costm = []
disp = 0.01
for i in range(10000):
    b = np.random.normal(prod, disp, (500, 1))
    x, cost, *_ = lstsq(A, b, method="ne")
    costm.append(cost)
costm = np.array(costm)

# chi_S = stats.chi2.fit(costm)
# print(chi_S)
# df_chi = chi_S[0]
# loc_chi = chi_S[1]
# scale_chi = chi_S[2]
# chi2 = stats.chi2.rvs(df=df_chi, loc=loc_chi, scale=scale_chi, size=len(costm))
# result_x = stats.ks_2samp(costm, chi2)
# print(result_x)

# постройка графика
plt.figure()  # новый график
y = stats.chi2.pdf(costm / disp ** 2, df=np.shape(A)[0] - np.shape(A)[1])
plt.plot(costm, y / disp ** 2, 'o', lw=5, alpha=0.6, label='chi2 pdf')
plt.hist(costm, 40, density=True, label="hist")
plt.xlabel('невязяка', fontsize=16)
plt.ylabel('плотность вероятности', fontsize=16)
plt.title('Частотная гистограмма \n величины невязки и теоретическое распределение')
plt.legend()
# plt.show()
plt.savefig('chi2.png')

if __name__ == "__main__":
    pass
