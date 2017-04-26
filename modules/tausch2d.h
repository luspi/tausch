/*!
 * \file
 * \author  Lukas Spies <LSpies@illinois.edu>
 * \version 1.0
 */

#ifndef TAUSCH2D_H
#define TAUSCH2D_H

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

/*!
 * Use real_t in code to allow easier switch between double/float.
 */
typedef double real_t;

/*!
 *
 * \brief
 *  A library providing a clean and efficient interface for halo exchange in two dimensions.
 *
 * %Tausch2D is a library that provides a clean and efficient C and C++ API for halo exchange for two dimensional structured grids that are split into a structured coarse mesh for MPI. It supports halo exchange across the partition boundaries, and across a CPU/GPU boundary for GPU partitions living centered inside a CPU partition.
 */
class Tausch2D {

public:

    /*!
     * These are the edges available for inter-MPI halo exchanges: LEFT, RIGHT, TOP, BOTTOM.
     */
    enum Edge { LEFT, RIGHT, TOP, BOTTOM };

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
     * \param cpuHaloWidth
     *  The width of the CPU-to-CPU halo, i.e., the inter-MPI halo.
     * \param comm
     *  The MPI Communictor to be used. %Tausch2D will duplicate the communicator, thus it is safe to have multiple instances of %Tausch2D working with the same communicator. By default, MPI_COMM_WORLD will be used.
     */
    explicit Tausch2D(int localDimX, int localDimY, int mpiNumX, int mpiNumY, int cpuHaloWidth, MPI_Comm comm = MPI_COMM_WORLD);

    /*!
     * Destructor freeing any allocated memory.
     */
    ~Tausch2D();

    /*!
     * Tells %Tausch2D where to find the buffer for the CPU data.
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
    void performCpuToCpu() { startCpuEdge(LEFT); startCpuEdge(RIGHT); completeCpuEdge(LEFT); completeCpuEdge(RIGHT);
                             startCpuEdge(TOP); startCpuEdge(BOTTOM); completeCpuEdge(TOP); completeCpuEdge(BOTTOM); }

    /*!
     * Start the inter-MPI halo exchange across the given edge.
     *
     * \param edge
     *  The edge across which the halo exchange is supposed to be started. It can be either one of the enum Tausch2D::LEFT, Tausch2D::RIGHT, Tausch2D::TOP, Tausch2D::BOTTOM.
     */
    void startCpuEdge(Edge edge);

    /*!
     * Completes the inter-MPI halo exchange across the given edge. This has to come *after* calling startCpuEdge() on the same edge.
     *
     * \param edge
     *  The edge across which the halo exchange is supposed to be completed. It can be either one of the enum Tausch2D::LEFT, Tausch2D::RIGHT, Tausch2D::TOP, Tausch2D::BOTTOM.
     */
    void completeCpuEdge(Edge edge);

    /*!
     * Get the MPI communicator that %Tausch2D uses for all of its communication. Be careful not to 'overwrite' MPI tags that %Tausch2D uses.
     *
     * \return
     *  Returns the MPI communicator used by %Tausch2D.
     */
    MPI_Comm getMPICommunicator() { return TAUSCH_COMM; }

#ifdef TAUSCH_OPENCL

    /*!
     * Enable OpenCL for the current %Tausch2D object. This causes %Tausch2D to set up its own OpenCL environment.
     *
     * Note: This is only available if %Tausch2D was compiled with OpenCL support!
     *
     * \param gpuHaloWidth
     *  The width of the CPU/GPU halo.
     * \param blockingSyncCpuGpu
     *  Whether to sync the CPU and GPU. This is necessary when running both parts in asynchronous threads, but causes a deadlock when only one thread is used. Default value: true
     * \param clLocalWorkgroupSize
     *  The local workgroup size for each kernel call. Default value: 64
     * \param giveOpenCLDeviceName
     *  Whether %Tausch2D should print out the OpenCL device name. This can come in handy for debugging. Default value: false
     */
    void enableOpenCL(int gpuHaloWidth, bool blockingSyncCpuGpu = true, int clLocalWorkgroupSize = 64, bool giveOpenCLDeviceName = false);

    /*!
     * Overloaded function. Enabled OpenCL for the current %Tausch2D object, making %Tausch2D use the user-provided OpenCL environment.
     *
     * Note: This is only available if %Tausch2D was compiled with OpenCL support!
     *
     * \param cl_defaultDevice
     *  The OpenCL device.
     * \param cl_context
     *  The OpenCL context
     * \param cl_queue
     *  The OpenCL queue
     * \param gpuHaloWidth
     *  The width of the CPU/GPU halo.
     * \param blockingSyncCpuGpu
     *  Whether to sync the CPU and GPU. This is necessary when running both parts in asynchronous threads, but causes a deadlock when only one thread is used. Default value: true
     * \param clLocalWorkgroupSize
     *  The local workgroup size for each kernel call. Default value: 64
     */
    void enableOpenCL(cl::Device &cl_defaultDevice, cl::Context &cl_context, cl::CommandQueue &cl_queue, int gpuHaloWidth, bool blockingSyncCpuGpu = true, int clLocalWorkgroupSize = 64);

    /*!
     * Tells %Tausch2D where to find the buffer for the GPU data and some of its main important details.
     *
     * Note: This is only available if %Tausch2D was compiled with OpenCL support!
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
     * Convenience function that calls the necessary functions performing a halo exchange from the CPU to GPU. It calls startCpuToGpu() and completeGpuToCpu().
     *
     * Note: This is only available if %Tausch2D was compiled with OpenCL support!
     */
    void performCpuToGpu() { startCpuToGpu(); completeGpuToCpu(); }

    /*!
     * Convenience function that calls the necessary functions performing a halo exchange across the MPI ranks and from the CPU to GPU. It interweaves MPI communication with write to shared memory for sending data to the GPU. The CPU/GPU methods called are startCpuToGpu() and completeGpuToCpu().
     *
     * Note: This is only available if %Tausch2D was compiled with OpenCL support!
     */
    void performCpuToCpuAndCpuToGpu() { startCpuEdge(LEFT); startCpuEdge(RIGHT); startCpuToGpu();
                                        completeCpuEdge(LEFT); completeCpuEdge(RIGHT); startCpuEdge(TOP); startCpuEdge(BOTTOM);
                                        completeGpuToCpu(); completeCpuEdge(TOP); completeCpuEdge(BOTTOM); }

    /*!
     * Convenience function that calls the necessary functions performing a halo exchange from the GPU to CPU. It calls startGpuToCpu() and completeCpuToGpu().
     *
     * Note: This is only available if %Tausch2D was compiled with OpenCL support!
     */
    void performGpuToCpu() { startGpuToCpu(); completeCpuToGpu(); }

    /*!
     * Start the halo exchange from the CPU to the GPU (called by the CPU thread). This writes the required halo data to shared memory using atomic operations.
     *
     * Note: This is only available if %Tausch2D was compiled with OpenCL support!
     */
    void startCpuToGpu();
    /*!
     * Start the halo exchange from the GPU to the CPU (called by te GPU thread). This downloads the required halo data from the GPU and writes it to shared memory using atomic operations.
     *
     * Note: This is only available if %Tausch2D was compiled with OpenCL support!
     */
    void startGpuToCpu();

    /*!
     * Completes the halo exchange from the CPU to the GPU (called by the GPU thread). This takes the halo data the GPU wrote to shared memory and loads it into the respective buffer positions. This has to come *after* calling startCpuToGpu().
     *
     * Note: This is only available if %Tausch2D was compiled with OpenCL support!
     */
    void completeCpuToGpu();
    /*!
     * Completes the halo exchange from the GPU to the CPU (called by the CPU thread). This takes the halo data the CPU wrote to shared memory and uploads it to the GPU. This has to come *after* calling startGpuToCpu().
     *
     * Note: This is only available if %Tausch2D was compiled with OpenCL support!
     */
    void completeGpuToCpu();

    /*!
     * Return the OpenCL context used by %Tausch2D. This can be especially useful when %Tausch2D uses its own OpenCL environment and the user wants to piggyback on that.
     *
     * Note: This is only available if %Tausch2D was compiled with OpenCL support!
     *
     * \return
     *  The OpenCL context used by %Tausch2D.
     */
    cl::Context getContext() { return cl_context; }
    /*!
     * Return the OpenCL command queue used by %Tausch2D. This can be especially useful when %Tausch2D uses its own OpenCL environment and the user wants to piggyback on that.
     *
     * Note: This is only available if %Tausch2D was compiled with OpenCL support!
     *
     * \return
     *  The OpenCL command queue used by %Tausch2D.
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
    int cpuHaloWidth;

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

    // The communicator in use by Tausch2D
    MPI_Comm TAUSCH_COMM;

    // Holding the MPI requests so we can call Wait on the right ones
    MPI_Request cpuToCpuSendRequest[4];
    MPI_Request cpuToCpuRecvRequest[4];

#ifdef TAUSCH_OPENCL

    // The OpenCL Device, Context and CommandQueue in use by Tausch2D (either set up by Tausch2D or passed on from user)
    cl::Device cl_defaultDevice;
    cl::Context cl_context;
    cl::CommandQueue cl_queue;

    int gpuHaloWidth;

    // The OpenCL buffer holding the data on the GPU
    cl::Buffer gpuData;

    // Some meta information about the OpenCL region
    int gpuDimX, gpuDimY;
    cl::Buffer cl_gpuDimX, cl_gpuDimY;
    cl::Buffer cl_gpuHaloWidth;

    // Methods to set up the OpenCL environment and compile the required kernels
    void setupOpenCL(bool giveOpenCLDeviceName);
    void compileKernels();

    // Pointers to atomic arrays for communicating halo data between CPU and GPU thread via shared memory
    std::atomic<real_t> *cpuToGpuBuffer;
    std::atomic<real_t> *gpuToCpuBuffer;

    // These are only needed/set when Tausch2D set up its own OpenCL environment, not needed otherwise
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

#endif // TAUSCH2D_H
