# C compiler
CXX = g++
ICC = icc
CXX_FLAGS = -g -fopenmp -O3 -lrt

bfs: bfs.cpp 
	$(CXX) $(CXX_FLAGS) bfs.cpp -o bfs 

bfs_offload: bfs.cpp
	$(ICC) $(CXX_FLAGS) -DOMP_OFFLOAD bfs.cpp -o bfs_offload

clean:
	rm -f bfs bfs_offload result.txt
