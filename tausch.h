/*!
 * \file
 * \author  Lukas Spies <LSpies@illinois.edu>
 * \version 1.0
 *
 * \section LICENSE
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of
 * the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details at
 * https://www.gnu.org/copyleft/gpl.html
 *
 * \mainpage
 * \section intro Introduction
 *
 * Tausch is a library that provides a clean and efficient C/C++ API for halo exchange for structured grids, split into a structured coarse mesh for MPI. It supports halo exchange across the partition boundaries, and across a CPU/GPU boundary for GPU partitions living perfectly centered inside a CPU partition. It comes with an API for two dimensions, Tausch2D, and an API for three dimensions, Tausch3D (work in progress).
 *
 * The interface is contained within a single header file \em tausch.h for both the C and C++ APIs. Nothing additional is required other than linking against the Tausch library. Both the C and C++ bindings are very similar. The underlying C++ API is documented here in detail, the C API works equivalently, with the object being called CTausch and any method having the prefix 'tausch_'.
 *
 * \em Note: Tausch requires C++11 support to work!
 *
 * \section mpipartitioning MPI partitioning
 *
 * Tausch assumes and takes advantage of a very structured layout of the MPI partitioning, as illustrated in the graph below, tough the dimensions of each MPI partition do not necessarily have to be the same. For example, MPI partition #2 and #4 can have a larger x dimension than MPI partition #1 and #3. The only important thing is that each shared edge between two MPI partitions has the same dimension on either MPI rank.
 *
 * \image html mpipartitioning.png width=500px
 * \image latex mpipartitioning.png width=500px
 *
 * \section gpupartitioning GPU subpartition
 *
 * The GPU subpartition (if present) is assumed to sit in the middle of the MPI partition. When passing on the x and y dimensions of the GPU partition, it has the same number of points on the CPU to its left/right and top/down. If the total number of points in either the x or y direction thare are to be hadled by the CPU is not even, then the right/top CPU part will handle one additional point (using the \em floor()/ceil() mathematical functions). The following graph illustrates this behavior.
 *
 * \em Note: Having a GPU partition is optional, Tausch also works with simple CPU-only partitions!
 *
 * \image html cpugpupartitioning.png width=500px
 * \image latex cpugpupartitioning.png width=500px
 *
 * \section example Example code
 *
 * The code below shows a simple example of a general use case of the Tausch library for a two dimensional domain. In order to use the OpenCL parts of Tausch, it, (1), needs to be compiled with OpenCL support and, (2), you need to define TAUSCH_OPENCL \em before including the tausch.h header! In this example, the halo exchange is done from two threads running asynchronously. Note that Tausch only uses MPI for CPU-to-CPU communication, the CPU/GPU halo exchange is done using atomic operations in shared memory. This is important to keep in mind for requesting the proper level of threading support with MPI_Init_thread().
 *
 * \code
 *
 * #include <future>
 *
 * #define TAUSCH_OPENCL
 * #include <tausch/tausch.h>
 *
 * // The two functions that will be executed asynchronously
 * void launchCPU();
 * void launchGPU();
 *
 * // This object is global, available from any one of the threads
 * Tausch2D *tausch;
 *
 * int main(int argc, char** argv) {
 *
 *     int provided;
 *     MPI_Init_thread(&argc,&argv,MPI_THREAD_SERIALIZED,&provided);
 *     // real life: do error check on level of available threading
 *
 *     // Get MPI info
 *     int mpiRank, mpiSize;
 *     MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
 *     MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
 *     int mpiNumX = std::sqrt(mpiSize);
 *     int mpiNumY = std::sqrt(mpiSize);
 *     // This has to match for our test
 *     if(mpiNumX*mpiNumY != mpiSize) {
 *         std::cout << "mpiDimX*mpiDimY != mpiSize" << std::endl;
 *         return 1;
 *     }
 *
 *     // the width of the halo we want to use
 *     int haloWidth = 1;
 *
 *     // the x and y dimension of the LOCAL MPI partition
 *     int localDimX = 100, localDimY = 100;
 *     // the dimensions of the GPU partition
 *     int gpuLocalDimX = 47, gpuLocalDimY = 62;
 *
 *     // Create the Tausch object
 *     tausch = new Tausch2D(localDimX, localDimY, mpiNumX, mpiNumY, haloWidth);
 *
 *     // Tell Tausch to set up its own OpenCL environment, blocking CPU/GPU sync (asynchronous threads), local workgroup size of 64, and don't tell us the OpenCL device names
 *     tausch->enableOpenCL(true, 64, false);
 *
 *     double *datCPU = new double[(localDimX+2*haloWidth)*(localDimY+2*haloWidth)]{};
 *     // omitted: filling array with actual values
 *
 *     // Create buffer and fill with zeros
 *     cl::Buffer datGPU;
 *     try {
 *         datGPU = cl::Buffer(tausch->getContext(), CL_MEM_READ_WRITE, (gpuLocalDimX+2*haloWidth)*(gpuLocalDimY+2*haloWidth)*sizeof(double));
 *         tausch->getQueue().enqueueFillBuffer(datGPU, 0, 0, (gpuLocalDimX+2*haloWidth)*(gpuLocalDimY+2*haloWidth)*sizeof(double));
 *     } catch(cl::Error error) {
 *         std::cout << error.what() << " (" << error.err() << ")" << std::endl;
 *         return 1;
 *     }
 *
 *     // Tell Tausch where to find the CPU and GPU data
 *     tausch->setCPUData(datCPU);
 *     tausch->setGPUData(datGPU, gpuLocalDimX, gpuLocalDimY);
 *
 *     // launch the GPU part in asynchronous thread
 *     std::future<void> thrdGPU(std::async(std::launch::async, &launchGPU));
 *     // launch the CPU part in the same thread
 *     launchCPU();
 *     // wait for everything to finish
 *     thrdGPU.wait();
 *
 *     // Clean up memory
 *     delete tausch;
 *
 *     // done!
 *     MPI_Finalize();
 *
 *     return 0;
 *
 * }
 *
 * void launchCPU() {
 *
 *     // we post the CPU received
 *     tausch->postCpuReceives();
 *
 *     // do some work...
 *
 *     std::cout << "CPU starting halo exchange..." << std::endl;
 *
 *     // exchange halos using convenience function. More fine-grained control possible
 *     tausch->performCpuToCpuAndCpuToGpu();
 *
 *     std::cout << "CPU done!" << std::endl;
 *
 * }
 * void launchGPU() {
 *
 *     // do some work...
 *
 *     std::cout << "GPU starting halo exchange..." << std::endl;
 *
 *     // exchange halos using convenience function. More fine-grained control possible
 *     tausch->performGpuToCpu();
 *
 *     std::cout << "GPU done!" << std::endl;
 *
 * }
 *
 * \endcode
 *
 * This code can be compiled with: `mpic++ examplecode.cpp -ltausch -lOpenCL -std=c++11 -o example`
 *
 */

#ifndef TAUSCH_H
#define TAUSCH_H

#ifdef __cplusplus
#include "modules/tausch2d.h"
#include "modules/tausch3d.h"
#else
#include "modules/ctausch2d.h"
#endif

#endif // TAUSCH_H
