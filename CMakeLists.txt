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
    src/fmmpm.cpp
    src/fmmpm.cu
    src/gravity_long.cpp
    src/gravity_short.cpp
    src/gravity_short.cu
    src/hpx.cpp
    src/initialize.cpp
    src/kick_pm.cpp
    src/kick_pme.cpp
    src/kick_pme.cu
    src/kick_treepm.cpp
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
	tigerpm/fixedcapvec.hpp
	tigerpm/fft.hpp
	tigerpm/fmmpm.hpp
	tigerpm/initialize.hpp
	tigerpm/kernels.hpp
	tigerpm/kick_pm.hpp
	tigerpm/kick_pme.hpp
	tigerpm/kick_treepm.hpp
	tigerpm/options.hpp
	tigerpm/particles.hpp
	tigerpm/range.hpp
	tigerpm/stack_vector.hpp
	tigerpm/test.hpp
	tigerpm/tigerpm.hpp
	tigerpm/timer.hpp
	tigerpm/tree.hpp
	tigerpm/util.hpp
)

execute_process(COMMAND cp ${PROJECT_SOURCE_DIR}/power.init .)


add_hpx_executable(
  tigerpm
  DEPENDENCIES
    Boost::boost cudart fftw3f gsl gslcblas
  SOURCES
    ${tigerpm_source_files}
  HEADERS
    ${tigerpm_header_files}
 )

add_executable( code_gen src/code_gen.cpp )

set_property(TARGET tigerpm PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(tigerpm PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_options(tigerpm PUBLIC "-DUSE_HPX")
target_compile_options(tigerpm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        --use_fast_math -G --maxrregcount 128 ;
                        >)
                        
set_property(TARGET tigerpm PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})


