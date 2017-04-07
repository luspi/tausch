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
 * Tausch is a library that provides a clean and efficient C/C++ API for halo exchange for structured grids, split into a structured coarse mesh for MPI. It supports halo exchange across the partition boundaries, and across a CPU/GPU boundary for GPU partitions living perfectly centered inside a CPU partition.
 *
 * The interface is contained within a single header file \em tausch.h for the C++ API and \em ctausch.h for the C API. Nothing additional is required other than linking against the Tausch library. Both the C and C++ bindings are very similar. The underlying C++ API is documented in detail, the C API works equivalently, with the object being called CTausch and any method having the prefix 'tausch_'.
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
 * The code below shows a simple example of a general use case of the Tausch library. In order to use the OpenCL parts of Tausch, it, (1), needs to be compiled with OpenCL support and, (2), you need to define TAUSCH_OPENCL \em before including the tausch.h header! In this example, the halo exchange is done from two threads running asynchronously. Note that Tausch only uses MPI for CPU-to-CPU communication, the CPU/GPU halo exchange is done using atomic operations in shared memory. This is important to keep in mind for requesting the proper level of threading support with MPI_Init_thread().
 *
 * \code
 *
 * #include <future>
 *
 * #define TAUSCH_OPENCL
 * #include <tausch.h>
 *
 * // The two functions that will be executed asynchronously
 * void launchCPU();
 * void launchGPU();
 *
 * // This object is global, available from any one of the threads
 * Tausch *tausch;
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
 *     tausch = new Tausch(localDimX, localDimY, mpiNumX, mpiNumY, haloWidth);
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

#ifndef _TAUSCH_H
#define _TAUSCH_H

#include <mpi.h>
#include <fstream>
#include <cmath>
#include <sstream>
#include <iostream>
#include <thread>
#include <future>

#ifdef TAUSCH_OPENCL
    #define __CL_ENABLE_EXCEPTIONS
    #include <CL/cl.hpp>
#endif

typedef double real_t;

/*!
 * \brief
 *  A library providing a clean and efficient interface for halo exchange.
 *
 * %Tausch is a library that provides a clean and efficient C and C++ API for halo exchange for structured grids, split into a structured coarse mesh for MPI. It supports halo exchange across the partition boundaries, and across a CPU/GPU boundary for GPU partitions living centered inside a CPU partition.
 */
class Tausch {

public:

    /*!
     * These are the edges available for inter-MPI halo exchanges: Left, Right, Top, Bottom.
     */
    enum Edge { Left = 0, Right, Top, Bottom };

    /*!
     *
     * The default class constructor, expecting some basic yet important details about the data.
     *
     * \param localDimX
     *  The x dimension of the local partition (not the global x dimension).
     * \param localDimY
     *  The y dimension of the local partition (not the global y dimension).
     * \param mpiNumX
     *  The number of MPI ranks lined up in the x direction. mpiNumX*mpiNumY has to be equal to the total number of MPI ranks.
     * \param mpiNumY
     *  The number of MPI ranks lined up in the y direction. mpiNumX*mpiNumY has to be equal to the total number of MPI ranks.
     * \param haloWidth
     *  The width of the halo between MPI ranks AND between the CPU/GPU (if applicable).
     * \param comm
     *  The MPI Communictor to be used. %Tausch will duplicate the communicator, thus it is safe to have multiple instances of %Tausch working with the same communicator. By default, MPI_COMM_WORLD will be used.
     */
    explicit Tausch(int localDimX, int localDimY, int mpiNumX, int mpiNumY, int haloWidth, MPI_Comm comm = MPI_COMM_WORLD);

    /*!
     * Destructor freeing any allocated memory.
     */
    ~Tausch();

    /*!
     * Tells %Tausch where to find the buffer for the CPU data.
     *
     * \param dat
     *  The buffer holding the CPU data. This is expected to be one contiguous buffer holding both the values owned by this MPI rank and the ghost values.
     */
    void setCPUData(real_t *dat);

    /*!
     * Post the MPI_Irecv required for the halo exchange. This has to be called before any halo exchange is started.
     */
    void postCpuReceives();

    /*!
     * Convenience function that calls the necessary functions performing a halo exchange across MPI ranks. First it calls a left/right halo exchange followed by a top/bottom halo exchange.
     */
    void performCpuToCpu() { startCpuEdge(Left); startCpuEdge(Right); completeCpuEdge(Left); completeCpuEdge(Right);
                             startCpuEdge(Top); startCpuEdge(Bottom); completeCpuEdge(Top); completeCpuEdge(Bottom); }

    /*!
     * Start the inter-MPI halo exchange across the given edge.
     *
     * \param edge
     *  The edge across which the halo exchange is supposed to be started. It can be either one of the enum Tausch::Left, Tausch::Right, Tausch::Top, Tausch::Bottom.
     */
    void startCpuEdge(Edge edge);

    /*!
     * Completes the inter-MPI halo exchange across the given edge. This has to come *after* calling startCpuEdge() on the same edge.
     *
     * \param edge
     *  The edge across which the halo exchange is supposed to be completed. It can be either one of the enum Tausch::Left, Tausch::Right, Tausch::Top, Tausch::Bottom.
     */
    void completeCpuEdge(Edge edge);

    /*!
     * Get the MPI communicator that %Tausch uses for all of its communication. Be careful not to 'overwrite' MPI tags that %Tausch uses.
     *
     * \return
     *  Returns the MPI communicator used by %Tausch.
     */
    MPI_Comm getMPICommunicator() { return TAUSCH_COMM; }

#ifdef TAUSCH_OPENCL

    /*!
     * Enable OpenCL for the current %Tausch object. This causes %Tausch to set up its own OpenCL environment.
     *
     * Note: This is only available if %Tausch was compiled with OpenCL support!
     *
     * \param blockingSyncCpuGpu
     *  Whether to sync the CPU and GPU. This is necessary when running both parts in asynchronous threads, but causes a deadlock when only one thread is used. Default value: true
     * \param clLocalWorkgroupSize
     *  The local workgroup size for each kernel call. Default value: 64
     * \param giveOpenCLDeviceName
     *  Whether %Tausch should print out the OpenCL device name. This can come in handy for debugging. Default value: false
     */
    void enableOpenCL(bool blockingSyncCpuGpu = true, int clLocalWorkgroupSize = 64, bool giveOpenCLDeviceName = false);

    /*!
     * Overloaded function. Enabled OpenCL for the current %Tausch object, making Tausch use the user-provided OpenCL environment.
     *
     * Note: This is only available if %Tausch was compiled with OpenCL support!
     *
     * \param cl_defaultDevice
     *  The OpenCL device.
     * \param cl_context
     *  The OpenCL context
     * \param cl_queue
     *  The OpenCL queue
     * \param blockingSyncCpuGpu
     *  Whether to sync the CPU and GPU. This is necessary when running both parts in asynchronous threads, but causes a deadlock when only one thread is used. Default value: true
     * \param clLocalWorkgroupSize
     *  The local workgroup size for each kernel call. Default value: 64
     */
    void enableOpenCL(cl::Device &cl_defaultDevice, cl::Context &cl_context, cl::CommandQueue &cl_queue, bool blockingSyncCpuGpu = true, int clLocalWorkgroupSize = 64);

    /*!
     * Tells %Tausch where to find the buffer for the GPU data and some of its main important details.
     *
     * Note: This is only available if %Tausch was compiled with OpenCL support!
     *
     * \param dat
     *  The OpenCL buffer holding the GPU data. This is expected to be one contiguous buffer holding both the values owned by this GPU and the ghost values.
     * \param gpuDimX
     *  The x dimension of the GPU buffer. This has to be less than localDimX-2*haloWidth.
     * \param gpuDimY
     *  The y dimension of the GPU buffer. This has to be less than localDimY-2*haloWidth.
     */
    void setGPUData(cl::Buffer &dat, int gpuDimX, int gpuDimY);

    /*!
     * Convenience function that calls the necessary functions performing a halo exchange from the CPU to GPU.
     *
     * Note: This is only available if %Tausch was compiled with OpenCL support!
     */
    void performCpuToGpu() { startCpuToGpu(); completeCpuToGpu(); }

    /*!
     * Convenience function that calls the necessary functions performing a halo exchange across the MPI ranks and from the CPU to GPU. It interweaves MPI communication with write to shared memory for sending data to the GPU.
     *
     * Note: This is only available if %Tausch was compiled with OpenCL support!
     */
    void performCpuToCpuAndCpuToGpu() { startCpuEdge(Left); startCpuEdge(Right); startCpuToGpu();
                                        completeCpuEdge(Left); completeCpuEdge(Right); startCpuEdge(Top); startCpuEdge(Bottom);
                                        completeCpuToGpu(); completeCpuEdge(Top); completeCpuEdge(Bottom); }

    /*!
     * Convenience function that calls the necessary functions performing a halo exchange from the GPU to CPU.
     *
     * Note: This is only available if %Tausch was compiled with OpenCL support!
     */
    void performGpuToCpu() { startGpuToCpu(); completeGpuToCpu(); }

    /*!
     * Start the halo exchange of the CPU looking at the GPU. This writes the required halo data to shared memory using atomic operations.
     *
     * Note: This is only available if %Tausch was compiled with OpenCL support!
     */
    void startCpuToGpu();
    /*!
     * Start the halo exchange of the GPU looking at the CPU. This downloads the required halo data from the GPU and writes it to shared memory using atomic operations.
     *
     * Note: This is only available if %Tausch was compiled with OpenCL support!
     */
    void startGpuToCpu();

    /*!
     * Completes the halo exchange of the CPU looking at the GPU. This takes the halo data the GPU wrote to shared memory and loads it into the respective buffer positions. This has to come *after* calling startCpuToGpu().
     *
     * Note: This is only available if %Tausch was compiled with OpenCL support!
     */
    void completeCpuToGpu();
    /*!
     * Completes the halo exchange of the GPU looking at the CPU. This takes the halo data the CPU wrote to shared memory and uploads it to the GPU. This has to come *after* calling startGpuToCpu().
     *
     * Note: This is only available if %Tausch was compiled with OpenCL support!
     */
    void completeGpuToCpu();

    /*!
     * Return the OpenCL context used by %Tausch. This can be especially useful when %Tausch uses its own OpenCL environment and the user wants to piggyback on that.
     *
     * Note: This is only available if %Tausch was compiled with OpenCL support!
     *
     * \return
     *  The OpenCL context used by %Tausch.
     */
    cl::Context getContext() { return cl_context; }
    /*!
     * Return the OpenCL command queue used by %Tausch. This can be especially useful when %Tausch uses its own OpenCL environment and the user wants to piggyback on that.
     *
     * Note: This is only available if %Tausch was compiled with OpenCL support!
     *
     * \return
     *  The OpenCL command queue used by %Tausch.
     */
    cl::CommandQueue getQueue() { return cl_queue; }

#endif

private:

    // The current MPI rank and size of TAUSCH_COMM world
    int mpiRank, mpiSize;
    // The number of MPI ranks in the x/y direction of the domain
    int mpiNumX, mpiNumY;

    // The x/y dimensions of the LOCAL partition
    int localDimX, localDimY;

    // Pointer to the CPU data
    real_t *cpuData;

    // The width of the halo
    int haloWidth;

    // Double pointer holding the MPI sent/recvd data across all edges
    real_t **cpuToCpuSendBuffer;
    real_t **cpuToCpuRecvBuffer;

    // Whether the necessary steps were taken before starting a halo exchange
    bool cpuInfoGiven;
    bool cpuRecvsPosted;

    // Which edge of the halo exchange has been started
    bool cpuStarted[4];

    // this refers to inter-partition boundaries
    bool haveBoundary[4];

    // The communicator in use by Tausch
    MPI_Comm TAUSCH_COMM;

    // Holding the MPI requests so we can call Wait on the right ones
    MPI_Request cpuToCpuSendRequest[4];
    MPI_Request cpuToCpuRecvRequest[4];

#ifdef TAUSCH_OPENCL

    // The OpenCL Device, Context and CommandQueue in use by Tausch (either set up by Tausch or passed on from user)
    cl::Device cl_defaultDevice;
    cl::Context cl_context;
    cl::CommandQueue cl_queue;

    // The OpenCL buffer holding the data on the GPU
    cl::Buffer gpuData;

    // Some meta information about the OpenCL region
    int gpuDimX, gpuDimY;
    cl::Buffer cl_gpuDimX, cl_gpuDimY;
    cl::Buffer cl_haloWidth;

    // Methods to set up the OpenCL environment and compile the required kernels
    void setupOpenCL(bool giveOpenCLDeviceName);
    void compileKernels();

    // Pointers to atomic arrays for communicating halo data between CPU and GPU thread via shared memory
    std::atomic<real_t> *cpuToGpuBuffer;
    std::atomic<real_t> *gpuToCpuBuffer;

    // These are only needed/set when Tausch set up its own OpenCL environment, not needed otherwise
    cl::Platform cl_platform;
    cl::Program cl_programs;

    // Collecting the halo data to be sent/recvd
    cl::Buffer cl_gpuToCpuBuffer;
    cl::Buffer cl_cpuToGpuBuffer;

    // Whether the necessary steps were taken before starting a halo exchange
    bool gpuEnabled;
    bool gpuInfoGiven;

    // Which halo exchange has been started
    bool cpuToGpuStarted;
    bool gpuToCpuStarted;

    // The local OpenCL workgroup size
    int cl_kernelLocalSize;

    // Function to synchronize the CPU and GPU thread. Only needed when using asynchronous computing (i.e., when blocking is enabled)
    void syncCpuAndGpu();
    std::atomic<int> sync_counter[2];
    std::atomic<int> sync_lock[2];

    // Whether there are two threads running and thus whether there needs to be blocking synchronisation happen
    bool blockingSyncCpuGpu;

#endif

};

#endif
