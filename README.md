# KluDo
Kernel Clustering based Protein Domain Assignment

## Prerequisites
It is recommended to run KluDo by a Python version of 3.6 or more, however there should not be any problem with running KluDo in any 3.X version. Also the following packages/programs should be installed before using KluDo:
* [numpy](https://numpy.org/) (Python)
* [scipy](https://www.scipy.org/) (Python)
* [biopython](https://biopython.org/) (Python)
* [igraph](https://igraph.org/python/) (Python)
* [sklearn](http://scikit-learn.github.io/stable) (Python)
* [prettytable](https://pypi.org/project/PrettyTable/) (Python)
* [DSSP](https://swift.cmbi.umcn.nl/gv/dssp/) (Linux/Windows/Mac binary executable)

## Usage
The examples here are based on Linux operating system. By using -help argument one can see a description of all arguments in KluDo:
```sh
python3 kludo.py -help
```
```
  -help                     Help
  -pdb [PATH]               PDB file Path (*)
  -chainid [ID]             Chain id (*)
  -dssppath [PATH]          DSSP binary file path (*)
  -numdomains [NUMBER]      The number of domains
  -minsegsize [SIZE]        Minimum segment size
  -mindomainsize [SIZE]     Minimum domain size
  -maxalphahelix [SIZE]     Maximum size of alpha-helix to contract
  -maxsegdomratio [RATIO]   Maximum ratio of segment count to domain count
  -kernel [TYPE]            The type of graph node kernel (**)
  -dispall                  Display all candidate partitionings
  -diffparamx [VALUE]       Diffusion parameter X (***)
  -diffparamy [VALUE]       Diffusion parameter Y (***)

 *   These arguments are necessary

 **  Should be choosen from the list: lap-exp-diff, markov-diff, reg-lap-diff and markov-exp-diff

 *** The parameters diffparamx and diffparamy are coefficient (x) and exponent (y) of node count (n),
     respectively, which determine the diffusion parameter, t, for each kernel. (t=xn^y)
```
Among these arguments three of them are necessary: -pdb, -chainid and -dssppath. If you don't pass values to the rest of arguments, default values are used for them. The default values are as follows:
* numdomains -> automatic
* minsegsize  -> 25
* mindomainsize -> 27
* maxalphahelix -> 30
* maxsegdomratio -> 1.6
* kernel -> lap-exp-diff
* diffparamx -> 0.0105
* diffparamy -> 1

Assume we have downloaded the file 1cid.pdb to the path ~/1cid.pdb and DSSP program is installed in the path /usr/bin/dssp. The following examples show how to run KluDo on Linux:

```sh
python3 kludo.py -pdb ~/1cid.pdb -chainid A -dssppath /usr/bin/dssp
```
