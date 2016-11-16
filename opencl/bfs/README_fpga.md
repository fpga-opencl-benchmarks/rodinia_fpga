# Kernel variations

See the github Wiki page for more general information. 

## v1

Straightforward single work-item kernel created by wrapping v0 in a
for loop from 0 to no_of_nodes and addin restrict.

## v2

Contributed by Hagiwara-san from Altera. Uses unrolling for BFS_1 kernel
and SIMD for BFS_2 kernel. More unrolling on BFS_1 kernel results in
worse performance.

## v3

Single work-item kernel created by unrolling loops as much as possible
without using any resources more than 100% on the de5net_a7 board.

## v5

Trying to resolve memory dependancy caused by g_cost[id]=g_cost[tid]+1
using a temp variable that loads g_cost[tid] from memory earlier in the
piepleine. The AOCL doesn't print anything about memory dependancy
anymore.
