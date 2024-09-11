# Lyapunov-Exponents
Part II Coding Project on calculating Lyapunov Exponents in Python

## Code File Structure
### Folder: Logistic Map
– LogisticMap.py: contains functions that perform the logistic map x(n) → x(n+1) on a 1D array of initial conditions, with options to vary r parameter for each initial condition and the number of iterations.

– LogisticCalcs.py: contains a function that calculates the Lyapunov Exponent of the logistic map

– LogisticAnalysis.py: Contains a variety of functions that test the LogisticCalcs algorithm and
produces plots

– BifurcationDiagram.py contains a function dedicated to creating a plot of the bifurcation diagram and corresponding Lyapunov exponents.

### Folder:Standard Map

– StandardMap.py: contains functions that perform the standard map xn → xn+1, also includes
code for generating random initial conditions and a grid of initial conditions.

– StandardMapCalcs.py: contains a function that calculates the spectrum of Lyapunov Exponents of the standard map

– StandardMapAnalysis.py: Contains a variety of functions that test the StandardMapCalcs algo- rithm and produces plots

– PhaseDiagram.py contains a function dedicated to creating phase diagrams for the Standard Map 

### Folder:Lorenz

– Lorenz.py: contains functions that return the derivatives for the lorenz equation and the associated derivatives for an infinitesimal perturbation, also contains functions to perform numerical integration on initial conditions and perturbation.

– LorenzCalcs.py: contains a function that calculates the largest Lyapunov Exponent of the Lorenz system.

– LorenzAnalysis.py: Contains a variety of functions that test the LorenzCalcs algorithm and produces plots

– LorenzAnimation.py contains a function dedicated to creating an animation of the Lorenz system. 
### Folder:QR
– QRDecomposition.py: contains an assortment of different algorithms for performing QR decomposition

– QRAnalysis.py: contains functions that analyse and compare the different QR algorithms, e.g. error and runtime
