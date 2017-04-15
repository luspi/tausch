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
 */

#ifndef TAUSCH3D_H
#define TAUSCH3D_H

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
 *
 * \class Tausch3D _tausch3d.h tausch/tausch.h
 *
 * \brief
 *  A library providing a clean and efficient interface for halo exchange in three dimensions.
 *
 * %Tausch3D is a library that provides a clean and efficient C and C++ API for halo exchange for three dimensional structured grids that are split into a structured coarse mesh for MPI. It supports halo exchange across the partition boundaries, and across a CPU/GPU boundary for GPU partitions living centered inside a CPU partition.
 */
class Tausch3D {

public:

    /*!
     * These are the edges available for inter-MPI halo exchanges: LEFT, RIGHT, TOP, BOTTOM, FRONT, BACK.
     */
    enum Edge { LEFT, RIGHT, TOP, BOTTOM, FRONT, BACK };

    //HERE HERE HERE

    /*!
     *
     * The default class constructor, expecting some basic yet important details about the data.
     *
     * \param localDimX
     *  The x dimension of the local partition (not the global x dimension).
     * \param localDimY
     *  The y dimension of the local partition (not the global y dimension).
     * \param localDimZ
     *  The z dimension of the local partition (not the global y dimension).
     * \param mpiNumX
     *  The number of MPI ranks lined up in the x direction. mpiNumX*mpiNumY*mpiNumZ has to be equal to the total number of MPI ranks.
     * \param mpiNumY
     *  The number of MPI ranks lined up in the y direction. mpiNumX*mpiNumY*mpiNumZ has to be equal to the total number of MPI ranks.
     * \param mpiNumZ
     *  The number of MPI ranks lined up in the y direction. mpiNumX*mpiNumY*mpiNumZ has to be equal to the total number of MPI ranks.
     * \param haloWidth
     *  The width of the halo between MPI ranks AND between the CPU/GPU (if applicable).
     * \param comm
     *  The MPI Communictor to be used. %Tausch3D will duplicate the communicator, thus it is safe to have multiple instances of %Tausch3D working with the same communicator. By default, MPI_COMM_WORLD will be used.
     */
    explicit Tausch3D(int localDimX, int localDimY, int localDimZ, int mpiNumX, int mpiNumY, int mpiNumZ, int haloWidth, MPI_Comm comm = MPI_COMM_WORLD);

    /*!
     * Destructor freeing any allocated memory.
     */
    ~Tausch3D();

    /*!
     * Tells %Tausch3D where to find the buffer for the CPU data.
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
     * Convenience function that calls the necessary functions performing a halo exchange across MPI ranks. First it calls a left/right halo exchange followed by a top/bottom halo exchange, finishing off by a front/back halo exchange
     */
    void performCpuToCpu() { startCpuEdge(LEFT); startCpuEdge(RIGHT); completeCpuEdge(LEFT); completeCpuEdge(RIGHT);
                             startCpuEdge(TOP); startCpuEdge(BOTTOM); completeCpuEdge(TOP); completeCpuEdge(BOTTOM);
                             startCpuEdge(FRONT); startCpuEdge(BACK); completeCpuEdge(FRONT); completeCpuEdge(BACK); }

    /*!
     * Start the inter-MPI halo exchange across the given edge.
     *
     * \param edge
     *  The edge across which the halo exchange is supposed to be started. It can be either one of the enum Tausch3D::LEFT, Tausch3D::RIGHT, Tausch3D::TOP, Tausch3D::BOTTOM, Tausch3D::FRONT, Tausch3D::BACK.
     */
    void startCpuEdge(Edge edge);

    /*!
     * Completes the inter-MPI halo exchange across the given edge. This has to come *after* calling startCpuEdge() on the same edge.
     *
     * \param edge
     *  The edge across which the halo exchange is supposed to be completed. It can be either one of the enum Tausch3D::LEFT, Tausch3D::RIGHT, Tausch3D::TOP, Tausch3D::BOTTOM, Tausch3D::FRONT, Tausch3D::BACK.
     */
    void completeCpuEdge(Edge edge);

    /*!
     * Get the MPI communicator that %Tausch3D uses for all of its communication. Be careful not to 'overwrite' MPI tags that %Tausch3D uses.
     *
     * \return
     *  Returns the MPI communicator used by %Tausch3D.
     */
    MPI_Comm getMPICommunicator() { return TAUSCH_COMM; }

#ifdef TAUSCH_OPENCL

    /*!
     * Enable OpenCL for the current %Tausch3D object. This causes %Tausch3D to set up its own OpenCL environment.
     *
     * Note: This is only available if %Tausch3D was compiled with OpenCL support!
     *
     * \param blockingSyncCpuGpu
     *  Whether to sync the CPU and GPU. This is necessary when running both parts in asynchronous threads, but causes a deadlock when only one thread is used. Default value: true
     * \param clLocalWorkgroupSize
     *  The local workgroup size for each kernel call. Default value: 64
     * \param giveOpenCLDeviceName
     *  Whether %Tausch3D should print out the OpenCL device name. This can come in handy for debugging. Default value: false
     */
    void enableOpenCL(bool blockingSyncCpuGpu = true, int clLocalWorkgroupSize = 64, bool giveOpenCLDeviceName = false);

    /*!
     * Overloaded function. Enabled OpenCL for the current %Tausch3D object, making %Tausch3D use the user-provided OpenCL environment.
     *
     * Note: This is only available if %Tausch3D was compiled with OpenCL support!
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
     * Tells %Tausch3D where to find the buffer for the GPU data and some of its main important details.
     *
     * Note: This is only available if %Tausch3D was compiled with OpenCL support!
     *
     * \param dat
     *  The OpenCL buffer holding the GPU data. This is expected to be one contiguous buffer holding both the values owned by this GPU and the ghost values.
     * \param gpuDimX
     *  The x dimension of the GPU buffer. This has to be less than localDimX-2*haloWidth.
     * \param gpuDimY
     *  The y dimension of the GPU buffer. This has to be less than localDimY-2*haloWidth.
     * \param gpuDimZ
     *  The z dimension of the GPU buffer. This has to be less than localDimZ-2*haloWidth.
     */
    void setGPUData(cl::Buffer &dat, int gpuDimX, int gpuDimY, int gpuDimZ);

    /*!
     * Convenience function that calls the necessary functions performing a halo exchange from the CPU to GPU.
     *
     * Note: This is only available if %Tausch3D was compiled with OpenCL support!
     */
    void performCpuToGpu() { startCpuToGpu(); completeGpuToCpu(); }

    /*!
     * Convenience function that calls the necessary functions performing a halo exchange across the MPI ranks and from the CPU to GPU. It interweaves MPI communication with write to shared memory for sending data to the GPU.
     *
     * Note: This is only available if %Tausch3D was compiled with OpenCL support!
     */
    void performCpuToCpuAndCpuToGpu() { startCpuEdge(LEFT); startCpuEdge(RIGHT); startCpuToGpu();
                                        completeCpuEdge(LEFT); completeCpuEdge(RIGHT); startCpuEdge(TOP); startCpuEdge(BOTTOM);
                                        completeGpuToCpu(); completeCpuEdge(TOP); completeCpuEdge(BOTTOM);
                                        startCpuEdge(FRONT); startCpuEdge(BACK); completeCpuEdge(FRONT); completeCpuEdge(BACK); }

    /*!
     * Convenience function that calls the necessary functions performing a halo exchange from the GPU to CPU.
     *
     * Note: This is only available if %Tausch3D was compiled with OpenCL support!
     */
    void performGpuToCpu() { startGpuToCpu(); completeCpuToGpu(); }

    /*!
     * Start the halo exchange of the CPU looking at the GPU. This writes the required halo data to shared memory using atomic operations.
     *
     * Note: This is only available if %Tausch3D was compiled with OpenCL support!
     */
    void startCpuToGpu();
    /*!
     * Start the halo exchange of the GPU looking at the CPU. This downloads the required halo data from the GPU and writes it to shared memory using atomic operations.
     *
     * Note: This is only available if %Tausch3D was compiled with OpenCL support!
     */
    void startGpuToCpu();

    /*!
     * Completes the halo exchange of the CPU looking at the GPU. This takes the halo data the GPU wrote to shared memory and loads it into the respective buffer positions. This has to come *after* calling startCpuToGpu().
     *
     * Note: This is only available if %Tausch3D was compiled with OpenCL support!
     */
    void completeCpuToGpu();
    /*!
     * Completes the halo exchange of the GPU looking at the CPU. This takes the halo data the CPU wrote to shared memory and uploads it to the GPU. This has to come *after* calling startGpuToCpu().
     *
     * Note: This is only available if %Tausch3D was compiled with OpenCL support!
     */
    void completeGpuToCpu();

    /*!
     * Return the OpenCL context used by %Tausch3D. This can be especially useful when %Tausch3D uses its own OpenCL environment and the user wants to piggyback on that.
     *
     * Note: This is only available if %Tausch3D was compiled with OpenCL support!
     *
     * \return
     *  The OpenCL context used by %Tausch3D.
     */
    cl::Context getContext() { return cl_context; }
    /*!
     * Return the OpenCL command queue used by %Tausch3D. This can be especially useful when %Tausch3D uses its own OpenCL environment and the user wants to piggyback on that.
     *
     * Note: This is only available if %Tausch3D was compiled with OpenCL support!
     *
     * \return
     *  The OpenCL command queue used by %Tausch3D.
     */
    cl::CommandQueue getQueue() { return cl_queue; }

#endif

private:

    // The current MPI rank and size of TAUSCH_COMM world
    int mpiRank, mpiSize;
    // The number of MPI ranks in the x/y direction of the domain
    int mpiNumX, mpiNumY, mpiNumZ;

    // The x/y dimensions of the LOCAL partition
    int localDimX, localDimY, localDimZ;

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
    bool cpuStarted[6];

    // this refers to inter-partition boundaries
    bool haveBoundary[6];

    // The communicator in use by Tausch3D
    MPI_Comm TAUSCH_COMM;

    // Holding the MPI requests so we can call Wait on the right ones
    MPI_Request cpuToCpuSendRequest[6];
    MPI_Request cpuToCpuRecvRequest[6];

#ifdef TAUSCH_OPENCL

    // The OpenCL Device, Context and CommandQueue in use by Tausch3D (either set up by Tausch3D or passed on from user)
    cl::Device cl_defaultDevice;
    cl::Context cl_context;
    cl::CommandQueue cl_queue;

    // The OpenCL buffer holding the data on the GPU
    cl::Buffer gpuData;

    // Some meta information about the OpenCL region
    int gpuDimX, gpuDimY, gpuDimZ;
    cl::Buffer cl_gpuDimX, cl_gpuDimY, cl_gpuDimZ;
    cl::Buffer cl_haloWidth;

    // Methods to set up the OpenCL environment and compile the required kernels
    void setupOpenCL(bool giveOpenCLDeviceName);
    void compileKernels();

    // Pointers to atomic arrays for communicating halo data between CPU and GPU thread via shared memory
    std::atomic<real_t> *cpuToGpuBuffer;
    std::atomic<real_t> *gpuToCpuBuffer;

    // These are only needed/set when Tausch3D set up its own OpenCL environment, not needed otherwise
    cl::Platform cl_platform;
    cl::Program cl_programs;

    // Collecting the halo data to be sent/recvd
    cl::Buffer cl_gpuToCpuBuffer;
    cl::Buffer cl_cpuToGpuBuffer;

    // Whether the necessary steps were taken before starting a halo exchange
    bool gpuEnabled;
    bool gpuInfoGiven;

    // Which halo exchange has been started
    std::atomic<bool> cpuToGpuStarted;
    std::atomic<bool> gpuToCpuStarted;

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

#endif // TAUSCH3D_H
