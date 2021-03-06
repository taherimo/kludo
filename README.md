# KluDo
KluDo (Kernel based Clustering for Protein Domain Assignment), is an automatic framework for protein domain assignment, which incorporates graph node kernels as an advanced similarity measure.


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
*  -clustering (The clustering method: "spectral" or "kernel-kmeans"; Default: "spectral")
*  -numdomains (The number of domains; Default: automatic)
*  -minsegsize (Minimum segment size; Default: 25)
*  -mindomainsize (Minimum domain size; Default: 27)
*  -maxalphahelix (Maximum size of alpha-helix to contract; Default: 30)
*  -maxsegdomratio (Maximum ratio of segment count to domain count; Default: 1.6)
*  -kernel (The type of graph node kernel; Default: lap-exp-diff)
*  -dispall (Display all candidate partitionings)
*  -diffparamx (Diffusion parameter X)
*  -diffparamy (Diffusion parameter Y)

Three of the arguments are mandatory: -pdb, -chainid and -dssppath. If you don't pass values to the rest of arguments, default values are utilized. Values of the arguments -diffparamx and -diffparamy should be passed simultaneously. These are coefficient (x) and exponent (y) of node count (n), respectively, which determine the diffusion parameter, t, for each kernel (t=xn^y). Default values of the arguments -diffparamx and -diffparam depend on the kernel type and clustering algorithm.

Also users can choose the Kernel type from the following options:
* lap-exp-diff (Laplacian Exponential Diffusion Kernel)
* markov-diff (Markov Diffusion Kernel)
* reg-lap-diff (Regularized Laplacian Diffusion Kernel)
* markov-exp-diff (Markov Exponential Diffusion Kernel)

Assume we have downloaded the file [1cid.pdb](https://files.rcsb.org/download/1CID.pdb) to the path ~/1cid.pdb and DSSP program is installed in the path /usr/bin/dssp. The following example shows how to run KluDo:

```sh
python3 kludo.py -pdb ~/1cid.pdb -chainid A -dssppath /usr/bin/dssp
```
## Web application
KluDo is also available as a web application at: [http://www.cbph.ir/tools/kludo](http://www.cbph.ir/tools/kludo)
