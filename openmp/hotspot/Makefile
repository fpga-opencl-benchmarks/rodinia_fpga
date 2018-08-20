include ../../common/make.config

ICC = icc
CXXFLAGS += -DOPEN

OFFLOAD_CC_FLAGS = -offload-option,mic,compiler,"-no-opt-prefetch"

hotspot: hotspot_openmp.cpp Makefile 
	$(CXX) $(CXXFLAGS) hotspot_openmp.cpp -o hotspot $(OPENMP_INC) -lrt

hotspot_offload: hotspot_openmp.cpp Makefile
	$(ICC) $(CC_FLAGS) $(OFFLOAD_CC_FLAGS) -DOMP_OFFLOAD hotspot_openmp.cpp -o hotspot_offload

clean:
	rm -f hotspot hotspot_offload
