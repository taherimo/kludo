# KluDo
KluDo (Diffusion Kernel-based Graph Node Clustering for Protein Domain Assignment), is an automatic framework that incorporates diffusion kernels on protein graphs as affinity measures between residues to decompose protein structures into structural domains.


## Prerequisites
It is recommended to run KluDo by a Python version of 3.6 or more, however there should be no problem with running KluDo in any 3.x version. Also the following packages/programs must be installed before using KluDo:
* [numpy](https://numpy.org/) (Python)
* [scipy](https://www.scipy.org/) (Python)
* [atomium](https://atomium.samireland.com/) (Python)
* [igraph](https://igraph.org/python/) (Python)
* [sklearn](http://scikit-learn.github.io/stable) (Python)
* [tslearn](https://tslearn.readthedocs.io/) (Python)
* [prettytable](https://pypi.org/project/PrettyTable/) (Python)
* [DSSP](https://swift.cmbi.umcn.nl/gv/dssp/) (Linux/Windows/Mac binary executable)

## Usage
The following commands are based on Linux operating system. By using --help argument one can see a description of all arguments in KluDo:
```sh
python3 kludo.py --help
```
The arguments are as follows:
*  --pdb (PDB file Path)
*  --chainid (Chain ID)
*  --dssppath (DSSP binary file path; Default: /usr/bin/dssp)
*  --clustering (The clustering method: "spectral" or "kernel-kmeans"; default: "spectral")
*  --numdomains (The number of domains; default: automatic)
*  --minsegsize (Minimum segment size; default: 25)
*  --mindomainsize (Minimum domain size; default: 27)
*  --maxalphahelix (Maximum size of alpha-helix to contract; default: 30)
*  --maxsegdomratio (Maximum ratio of segment count to domain count; default: 1.6)
*  --kernel (The type of graph node kernel; default: lap-exp-diff)
*  --dispall (Display all candidate partitionings)
*  --bw_x (Bandwidth parameter x)
*  --bw_y (Bandwidth parameter y)

Two of the arguments are mandatory: --pdb and --chainid. If you don't pass values to the rest of arguments, default values are used. Values of the arguments --bw_x and --bw_y should be passed simultaneously. These are coefficient (x) and exponent (y) of the node count (n), respectively, which determine the bandwidth parameter (β or t) for each kernel (t=xn^y). Default values of the arguments --bw_x and --bw_y depend on the kernel type and the clustering algorithm.

Fore the argument --kernel users can choose one of the following options:
* markov-diff (Markov Diffusion Kernel, default)
* lap-exp-diff (Laplacian Exponential Diffusion Kernel)
* reg-lap-diff (Regularized Laplacian Diffusion Kernel)
* markov-exp-diff (Markov Exponential Diffusion Kernel)

Also for the argument --clustering there are following options:
* spectral (default)
* kernel-kmeans

As an example assuming that the pdb file [1cid.pdb](https://files.rcsb.org/download/1CID.pdb) is stored in the path ~/1cid.pdb the following command runs KluDo with the minimal arguments:

```sh
python3 kludo.py --pdb ~/1cid.pdb --chainid A
```
## Web application
KluDo is also available as a web application at: [http://www.cbph.ir/tools/kludo](http://www.cbph.ir/tools/kludo)
