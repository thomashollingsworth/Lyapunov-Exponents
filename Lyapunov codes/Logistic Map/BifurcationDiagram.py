"""Plot a Bifurcation diagram for the Logistic Map and Compare with Lyapunov Exponent"""

import numpy as np
import matplotlib.pyplot as plt
import LogisticCalcs as logistic

# Plotting the bifurcation diagram and Lyapunov exponents

r_values = np.linspace(
    0.01, 4, 1000
)  # Range of 1000 r values between 0.001 and 4 (avoids log(0) errors)
x0 = 0.5 * np.ones(1000)  # All calcs start from an x0 value of 0.5

plt.subplot(2, 1, 1)
plt.plot(
    r_values,
    logistic.logistic_lyapunov(1000, 1000, 300, rvals=r_values, x0vals=x0)[0],
    "k,",
)
plt.xlabel("$\it{r}$")
plt.xlim(0.5, 4)
plt.ylabel("$\it{Equilibrium}$ $\it{values}$")
ax1 = plt.gca()
ax1.spines[["top", "right"]].set_visible(False)
plt.suptitle("Bifurcation Diagram and Lyapunov Exponents for Logistic Map")
ax1.axvline(
    x=3.56995,
    ymin=-1.2,
    ymax=1,
    c="red",
    linewidth=0.5,
    linestyle="dashed",
    zorder=0,
    clip_on=False,
)
ax1.axvline(
    x=1,
    ymin=-1.2,
    ymax=1,
    c="red",
    linewidth=0.5,
    linestyle="dashed",
    zorder=0,
    clip_on=False,
)
ax1.axvline(
    x=3,
    ymin=-1.2,
    ymax=1,
    c="red",
    linewidth=0.5,
    linestyle="dashed",
    zorder=0,
    clip_on=False,
)

plt.subplot(2, 1, 2)
plt.plot(
    r_values,
    logistic.logistic_lyapunov(1000, 1000, 300, rvals=r_values, x0vals=x0)[1],
    "b",
    linewidth=0.75,
)
ax2 = plt.gca()
ax2.spines["bottom"].set_position("zero")
ax2.spines[["top", "right"]].set_visible(False)

plt.ylabel("$\it{Lyapunov}$ $\it{exponent}$")
plt.xlim(0.5, 4)
plt.xlabel("$\it{r}$")
plt.xticks(color="w")
plt.ylim(-2, 1)
ax2.axvline(
    x=3.56995,
    ymin=0,
    ymax=1.2,
    c="red",
    linewidth=0.5,
    linestyle="dashed",
    zorder=0,
    clip_on=False,
)
ax2.axvline(
    x=1,
    ymin=0,
    ymax=1.2,
    c="red",
    linewidth=0.5,
    linestyle="dashed",
    zorder=0,
    clip_on=False,
)
ax2.axvline(
    x=3,
    ymin=0,
    ymax=1.2,
    c="red",
    linewidth=0.5,
    linestyle="dashed",
    zorder=0,
    clip_on=False,
)
plt.tight_layout()
# plt.savefig("LogisticMapBifurcation.png", dpi=200)
plt.show()
