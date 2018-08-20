# Rodinia Benchmark Suite for OpenCL-based FPGAs

The Rodinia Benchmark Suite is a set of benchmarks originally developed at University of Virginia. See `README_original` for the original description, or visit [here](https://www.cs.virginia.edu/~skadron/wiki/rodinia/index.php) for more details.

The Rodinia Benchmark Suite for OpenCL-based FPGAs is our modified version of the original benchmarks for FPGAs using Intel FPGA SDK for OpenCL. Xilinx FPGAs are NOT supported. The following benchmarks are ported to Intel FPGAs:

- nw (full optimization)
- hotspot (full optimization)
- hotspot 3D (full optimization)
- pathfinder (full optimization)
- srad (full optimization)
- lud (full optimization)
- cfd
- bfs
- b+tree
- backprop
- lavaMD

Each modified benchmark is available under [the opencl directory](opencl). See [the original README file](README_original) for more details about each benchmark. Each FPGA version has a readme included that describes the parameters and optimizations for that version.

The input data files for the benchmarks are not included in this distribution and needs to be separately downloaded from [here](https://www.cs.virginia.edu/~skadron/wiki/rodinia/index.php).

## Important note

Makefiles are NOT guaranteed to work correctly for kernel compilation and should only be used for compiling the host codes using the HOST_ONLY=1 flag. Compile kernels manually with the settings reported in Hamid's PhD thesis.

## Publications

- Hamid Reza Zohouri, Naoya Maruyama, Aaron Smith, Motohiko Matsuda, and Satoshi Matsuoka, "Evaluating and Optimizing OpenCL Kernels for High Performance Computing with FPGAs," Proceedings of the ACM/IEEE International Conference for High Performance Computing, Networking, Storage and Analysis (SC'16), Nov. 2016. [Paper](https://dl.acm.org/citation.cfm?id=3014951)
- Artur Podobas, Hamid Reza Zohouri, Naoya Maruyama, Satoshi Matsuoka, "Evaluating High-Level Design Strategies on FPGAs for High-Performance Computing," Proceedings of the 27th International Conference on Field Programmable Logic and Applications (FPL'17), Sep. 2017. [Paper](https://ieeexplore.ieee.org/abstract/document/8056760/)
- Hamid Reza Zohouri, Artur Podobas, Satoshi Matsuoka, "Combined Spatial and Temporal Blocking for High-Performance Stencil Computation on FPGAs Using OpenCL," Proceedings of the 2018 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays (FPGA'18), Feb. 2018. [Paper](https://dl.acm.org/citation.cfm?id=3174248)
- Hamid Reza Zohouri, "High Performance Computing with FPGAs and OpenCL," PhD thesis, Tokyo Institute of Technology, Tokyo, Japan, Aug. 2018

## Contact

Hamid Reza Zohouri <br />
Tokyo Institute of Technology <br />
zohouri.h.aa@m.titech.ac.jp <br />
http://github.com/zohourih

Naoya Maruyama <br />
RIKEN Advanced Institute for Computational Science / Tokyo Institute of Technology <br />
nmaruyama@riken.jp <br />
http://github.com/naoyam <br />
http://mt.aics.riken.jp/~nmaruyama/
