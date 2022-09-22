#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def dummyplot():
    plt.plot(np.random.randint(10, size=5), np.random.randint(10, size=5))


plt1 = dummyplot()
plt2 = dummyplot()
pp = PdfPages("foo.pdf")
for i in [plt1, plt2]:
    pp.savefig(i)
pp.close()
