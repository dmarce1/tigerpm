cmake_minimum_required(VERSION 3.12.4)
cmake_policy(VERSION 3.12.4)
cmake_policy(SET CMP0003 NEW)
cmake_policy(SET CMP0074 NEW)
project(tigerpm CXX C)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
find_package(HPX REQUIRED NO_CMAKE_PACKAGE_REGISTRY)
find_package(Boost REQUIRED)
enable_language(CUDA)
set(CMAKE_CXX_FLAGS "-ffast-math -march=native -Wno-deprecated-declarations -Wno-format-security")
execute_process(COMMAND nvcc -lcuda ${PROJECT_SOURCE_DIR}/src/cuda_detect.cu -o cuda_detect)
execute_process(COMMAND ./cuda_detect OUTPUT_VARIABLE CUDA_ARCH)
message(STATUS "CUDA Architecture: ${CUDA_ARCH}")


set(tigerpm_source_files
	src/chainmesh.cpp
    src/cuda.cpp
    src/fft.cpp
    src/gravity_long.cpp
    src/gravity_short.cpp
    src/gravity_short.cu
    src/hpx.cpp
    src/kick_pm.cpp
    src/kick_pme.cpp
    src/kick_pme.cu
    src/kick_treepm.cu
    src/kick_treepm.cu
    src/main.cpp
    src/options.cpp
    src/particles.cpp
    src/util.cpp
    src/test.cpp
    src/tree.cpp
)      
                       
                        
                        

set(tigerpm_header_files
	tigerpm/chainmesh.hpp
	tigerpm/containers.hpp
	tigerpm/complex.hpp
	tigerpm/cuda.hpp
	tigerpm/gravity_long.hpp
	tigerpm/gravity_short.hpp
	tigerpm/hpx.hpp
	tigerpm/fixed.hpp
	tigerpm/fft.hpp
	tigerpm/kick_pm.hpp
	tigerpm/kick_pme.hpp
	tigerpm/kick_treepm.hpp
	tigerpm/options.hpp
	tigerpm/particles.hpp
	tigerpm/range.hpp
	tigerpm/test.hpp
	tigerpm/tigerpm.hpp
	tigerpm/timer.hpp
	tigerpm/tree.hpp
	tigerpm/util.hpp
)



add_hpx_executable(
  tigerpm
  DEPENDENCIES
    Boost::boost cudart fftw3f gsl gslcblas
  SOURCES
    ${tigerpm_source_files}
  HEADERS
    ${tigerpm_header_files}
 )


set_property(TARGET tigerpm PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(tigerpm PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_options(tigerpm PUBLIC "-DUSE_HPX")
target_compile_options(tigerpm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        --use_fast_math ;
                        >)
set_source_files_properties(src/kick_pme.cu PROPERTIES COMPILE_FLAGS "--maxrregcount 64")
                        
set_property(TARGET tigerpm PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})




#add_library(hpx_lite 
#	hpx_lite/src/hpx.cpp
#	hpx_lite/src/id_type.cpp
#	hpx_lite/src/thread.cpp
#	hpx_lite/src/future.cpp
#	hpx_lite/src/mutex.cpp
#	hpx_lite/src/serialize.cpp
#)


#add_executable(
#  tigerpm_lite
#  ${tigerpm_source_files}
#  ${tigerpm_header_files}
# )

#target_link_libraries(tigerpm_lite PUBLIC  boost_program_options cudart fftw3f gsl gslcblas hpx_lite hwloc)
#target_include_directories(tigerpm_lite PUBLIC ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/hpx_lite)
#target_include_directories(hpx_lite PUBLIC  ${PROJECT_SOURCE_DIR}/hpx_lite)
#target_compile_options(tigerpm_lite PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#                        --use_fast_math --maxrregcount 128;
#                        >)
#set_property(TARGET tigerpm_lite PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
