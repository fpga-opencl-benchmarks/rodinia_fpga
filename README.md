# Rodinia Benchmark Suite for OpenCL-based FPGAs

The Rodinia Benchmark Suite is a set of benchmarks originally developed at University of Virginia. See `README_original` for the original benchmarks, or visit [here](https://www.cs.virginia.edu/~skadron/wiki/rodinia/index.php) for more details.

The Rodinia Benchmark Suite for OpenCL-based FPGAs is our modified version of the original benchmarks for FPGAs using OpenCL. As of now, only the following benchmarks are ported to Altera FPGAs.

- nw
- hotspot
- pathfinder
- cfd
- srad
- lud

Each modified benchmark is available under [the opencl directory](opencl). See [the original README file](README_original) for more details about each benchmark.

The input data files for the benchmarks are not included in this distribution and needs to be separately downloaded from [here](https://www.cs.virginia.edu/~skadron/wiki/rodinia/index.php).

## Publication

- Hamid Reza Zohouri, Naoya Maruyama, Aaron Smith, Motohiko Matsuda, and Satoshi Matsuoka, "Evaluating and Optimizing OpenCL Kernels for High Performance Computing with FPGAs," Proceedings of the ACM/IEEE International Conference for High Performance Computing, Networking, Storage and Analysis (SC'16), Nov 2016. [Paper](http://dl.acm.org/ft_gateway.cfm?id=3014951&ftid=1810066&dwn=1&CFID=863386528&CFTOKEN=11866610)

## Contact

Naoya Maruyama <br />
RIKEN Advanced Institute for Computational Science / Tokyo Institute of Technology <br />
nmaruyama@riken.jp <br />
http://github.com/naoyam <br />
http://mt.aics.riken.jp/~nmaruyama/

Hamid Reza Zohouri <br />
Tokyo Institute of Technology <br />
zohouri.h.aa@m.titech.ac.jp <br />
http://github.com/zohourih
