from astropy.io import fits
from lsp import lstsq
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt

with fits.open('ccd.fits') as fits_file:
    hdu = fits_file[0]
    data = hdu.data
# часть кода подсказана семинаристом
data = data.astype(np.float32)
x = np.mean(data, axis=(1, 2, 3))
x = x - x[0]
sigmax = np.var(np.diff(data, axis=1), axis=(2, 3))
xne = np.hstack((x.reshape((-1, 1)), np.ones((100, 1))))

# определение коэфицентов линейной регрессии
a, cost, dis = lstsq(xne, sigmax, method="ne")
alph = a[0][0]
bet = a[1][0]
errora = dis[0][0] ** 0.5
errorb = dis[1][1] ** 0.5

# постройка зависимости sigma x (x)
plt.figure()
plt.plot(x, sigmax, "o", label='scatter', markersize=5.1)
plt.plot(x, alph * x + bet, label="y={:.1f}x+{:.1e}".format(alph, bet))
plt.fill_between(x, (alph - errora) * x + bet - errorb, (alph + errora) * x + bet + errorb, label='error', color="g")
# plt.errorbar(x, a[0] * x + a[1], xerr=errorx, yerr=errory, label='erory', fmt='o-', ecolor='red', markersize=1)
plt.xlabel('x', fontsize=16)
plt.ylabel('sigma x', fontsize=16)
plt.title("Зависимость дисперсии от x ")
plt.legend()
plt.grid()
plt.savefig('ccd.png')

g = 2 / alph
ron = (bet / 2) ** 0.5 * abs(g)
epsg = errora / alph
g_err = g * epsg
epsron = abs(epsg) + errorb / (2 * bet)
ron_err = (bet * g_err ** 2 / 2 + g ** 2 / (8 * bet) * dis[1][1]) ** 0.5
d = {
    "ron": round(ron, 2),
    "ron_err": round(ron_err, 2),
    "gain": round(g, 4),
    "gain_err": round(g_err, 4)
}

with open('ccd.json', 'w') as f:
    json.dump(d, f, indent=2)

if __name__ == "__main__":
    pass
