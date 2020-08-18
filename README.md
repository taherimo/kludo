# KluDo
Kernel Clustering based Protein Domain Assignment

## Prerequisites
It is recommended to run KluDo by a Python version of 3.6 or more, however there should be no problem with running KluDo in any 3.x version. Also the following packages/programs must be installed before using KluDo:
* [numpy](https://numpy.org/) (Python)
* [scipy](https://www.scipy.org/) (Python)
* [biopython](https://biopython.org/) (Python)
* [igraph](https://igraph.org/python/) (Python)
* [sklearn](http://scikit-learn.github.io/stable) (Python)
* [prettytable](https://pypi.org/project/PrettyTable/) (Python)
* [DSSP](https://swift.cmbi.umcn.nl/gv/dssp/) (Linux/Windows/Mac binary executable)

## Usage
The following commands are based on Linux operating system. By using -help argument one can see a description of all arguments in KluDo:
```sh
python3 kludo.py -help
```
The arguments are as follows:
*  -pdb (PDB file Path)
*  -chainid
*  -dssppath (DSSP binary file path)
*  -numdomains (The number of domains)
*  -minsegsize (Minimum segment size)
*  -mindomainsize (Minimum domain size)
*  -maxalphahelix (Maximum size of alpha-helix to contract)
*  -maxsegdomratio (Maximum ratio of segment count to domain count)
*  -kernel (The type of graph node kernel)
*  -dispall (Display all candidate partitionings)
*  -diffparamx (Diffusion parameter X)
*  -diffparamy (Diffusion parameter Y)

Three of the arguments are mandatory: -pdb, -chainid and -dssppath. If you don't pass values to the rest of arguments, default values are utilized. Values of the arguments -diffparamx and -diffparamy should be passed simultaneously. The parameters diffparamx and diffparamy are coefficient (x) and exponent (y) of node count (n), respectively, which determine the diffusion parameter, t, for each kernel (t=xn^y). Moreover the user can choose the Kernel type from following options:
* lap-exp-diff (Laplacian Exponential Diffusion Kernel)
* markov-diff (Markov Diffusion Kernel)
* reg-lap-diff (Regularized Laplacian Diffusion Kernel)
* markov-exp-diff (Markov Exponential Diffusion Kernel)

The default values of the parameters are as follows:
* numdomains -> automatic
* minsegsize  -> 25
* mindomainsize -> 27
* maxalphahelix -> 30
* maxsegdomratio -> 1.6
* kernel -> lap-exp-diff
* diffparamx -> lap-exp-diff: 0.0105, markov-diff: 0.1024, reg-lap-diff: 0.00005, markov-exp-diff: 0.005
* diffparamy -> 1

Assume we have downloaded the file 1cid.pdb to the path ~/1cid.pdb and DSSP program is installed in the path /usr/bin/dssp. The following examples show how to run KluDo on Linux:

```sh
python3 kludo.py -pdb ~/1cid.pdb -chainid A -dssppath /usr/bin/dssp
```
