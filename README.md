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
$ python3 kludo.py --help
```
The arguments are as follows:
*  --pdb (PDB file Path)
*  --chainid (Chain ID)
*  --dssppath (DSSP binary file path; Default: /usr/bin/dssp)
*  --clustering (The clustering method: "spectral" or "KK"; default: "SP")
*  --numdomains (The number of domains; default: automatic)
*  --minsegsize (Minimum segment size; default: 27)
*  --mindomainsize (Minimum domain size; default: 27)
*  --maxalphahelix (Maximum size of alpha-helix to contract; default: 30)
*  --maxsegdomratio (Maximum ratio of segment count to domain count; default: 1.5)
*  --kernel (The type of graph node kernel; default: LED)
*  --dispall (Display all candidate partitionings)
*  --bw_a (coefficient of bandwidth coefficient)
*  --bw_b (exponent of radius of gyration to calculate bandwidth parameter)

Two of the arguments are mandatory: --pdb and --chainid. If you don't pass values to the rest of arguments, default values are used. Values of the arguments --bw_a and --bw_b should be passed simultaneously. These are coefficient (a) and exponent (b) of the Rg, respectively, which determine the bandwidth parameter (β or t) for each kernel (a * Rg ^ b). Default values of these arguments depend on the kernel type and the clustering algorithm.

Fore the argument --kernel users can choose one of the following options:
* LED (Laplacian Exponential Diffusion Kernel)
* MD (Markov Diffusion Kernel, default)
* MED (Markov Exponential Diffusion Kernel)
* RL (Regularized Laplacian Diffusion Kernel)

Also for the argument --clustering there are following options:
* SP (spectral, default)
* KK (kernel k-means)

As an example assuming that the pdb file [1cid.pdb](https://files.rcsb.org/download/1CID.pdb) is stored in the path ~/1cid.pdb the following command runs KluDo with the minimal arguments:

```sh
$ python3 kludo.py --pdb ~/1cid.pdb --chainid A
```
## Web application
KluDo is also available as a web application at: [http://www.cbph.ir/tools/kludo](http://www.cbph.ir/tools/kludo)
