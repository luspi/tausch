/*!
 * @file
 * @author  Lukas Spies <LSpies@illinois.edu>
 * @version 1.0
 *
 * @section LICENSE
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
 * @section DESCRIPTION
 *
 * The Q2Matrix class holds the data (nodes, x-edges, y-edges, cell-centers) - corresponding to rows - and their interaction with the surrounding points - corresponding to columns. The Q2Matrix stores the values of the Laplacian part of the system matrix.
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

#ifdef OPENCL
    #define __CL_ENABLE_EXCEPTIONS
    #include <CL/cl.hpp>
#endif

typedef double real_t;

/*!
 * @brief
 *  A library providing a clean and efficient interface for halo exchange.
 *
 * %Tausch is a library that provides a clean and efficient C and C++ API for halo exchange for structured grids, split into a structured coarse mesh for MPI. It supports halo exchange across the partition boundaries, and across a CPU/GPU boundary for GPU partitions living perfectly centered inside a CPU partition.
 */
class Tausch {

    enum { Left = 0, Right, Top, Bottom };
    typedef int Edge;

public:
    /**
     * @brief
     *  The class constructor, taking a few necessary default variables.
     * @param localDimX
     *  The x dimension of the local partition (not the global x dimension).
     * @param localDimY
     *  The y dimension of the local partition (not the global y dimension).
     * @param mpiNumX
     *  The number of MPI ranks lined up in the x direction. mpiNumX*mpiNumY has to be equal to the total number of MPI ranks.
     * @param mpiNumY
     *  The number of MPI ranks lined up in the y direction. mpiNumX*mpiNumY has to be equal to the total number of MPI ranks.
     */
    explicit Tausch(int localDimX, int localDimY, int mpiNumX, int mpiNumY, MPI_Comm comm = MPI_COMM_WORLD);
    ~Tausch();

    void postCpuReceives();

    void performCpuToCpu() { startCpuEdge(Left); startCpuEdge(Right); completeCpuEdge(Left); completeCpuEdge(Right);
                             startCpuEdge(Top); startCpuEdge(Bottom); completeCpuEdge(Top); completeCpuEdge(Bottom); }

    void startCpuEdge(Edge edge);
    void completeCpuEdge(Edge edge);

    MPI_Comm getMPICommunicator() { return TAUSCH_COMM; }

    void setHaloWidth(int haloWidth) { this->haloWidth = 1; }
    void setCPUData(real_t *dat);

#ifdef OPENCL

    void enableOpenCL(bool blockingSyncCpuGpu, bool setupOpenCL = false, int clLocalWorkgroupSize = 64, bool giveOpenCLDeviceName = false);
    void enableOpenCL(cl::Device &cl_defaultDevice, cl::Context &cl_context, cl::CommandQueue &cl_queue, bool blockingSyncCpuGpu, int clLocalWorkgroupSize = 64);

    void performCpuToGpu() { startCpuToGpu(); completeCpuToGpu(); }

    void performCpuToCpuAndCpuToGpu() { startCpuEdge(Left); startCpuEdge(Right); startCpuToGpu();
                                        completeCpuEdge(Left); completeCpuEdge(Right); startCpuEdge(Top); startCpuEdge(Bottom);
                                        completeCpuToGpu(); completeCpuEdge(Top); completeCpuEdge(Bottom); }

    void performGpuToCpu() { startGpuToCpu(); completeGpuToCpu(); }

    void startCpuToGpu();
    void startGpuToCpu();

    void completeCpuToGpu();
    void completeGpuToCpu();

    void setGPUData(cl::Buffer &dat, int gpuDimX, int gpuDimY);
    bool isGpuEnabled() { return gpuEnabled; }

    cl::Context cl_context;
    cl::CommandQueue cl_queue;

    void checkOpenCLError(cl_int clErr, std::string loc);

    void syncCpuAndGpu();

#endif

private:
    void EveryoneOutput(const std::string &inMessage);

    int localDimX, localDimY;
    real_t *cpuData;

    int haloWidth;

    int mpiRank, mpiSize;
    int mpiNumX, mpiNumY;

    real_t **cpuToCpuSendBuffer;
    real_t **cpuToCpuRecvBuffer;

    bool cpuInfoGiven;
    bool cpuRecvsPosted;

    bool cpuStarted[4];

    // this refers to inter-partition boundaries
    bool haveBoundary[4];

    MPI_Comm TAUSCH_COMM;

    MPI_Request cpuToCpuSendRequest[4];
    MPI_Request cpuToCpuRecvRequest[4];

#ifdef OPENCL

    cl::Buffer gpuData;

    int gpuDimX, gpuDimY;
    cl::Buffer cl_gpuDimX, cl_gpuDimY;

    void setupOpenCL(bool giveOpenCLDeviceName);
    void compileKernels();

    std::atomic<real_t> *cpuToGpuBuffer;
    std::atomic<real_t> *gpuToCpuBuffer;

    cl::Platform cl_platform;
    cl::Device cl_defaultDevice;
    cl::Program cl_programs;

    cl::Buffer cl_gpuToCpuBuffer;
    cl::Buffer cl_cpuToGpuBuffer;

    bool gpuEnabled;
    bool gpuInfoGiven;

    bool cpuToGpuStarted;
    bool gpuToCpuStarted;

    int cl_kernelLocalSize;

    std::atomic<int> sync_counter[2];
    std::atomic<int> sync_lock[2];

    bool blockingSyncCpuGpu;
#endif

};

#endif
