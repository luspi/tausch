#ifndef _TAUSCH_H
#define _TAUSCH_H

#include <mpi.h>
#include <fstream>
#include <cmath>
#include <sstream>
#include <iostream>
#include <thread>
#include <future>
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

typedef double real_t;

class Tausch {

    enum { Left = 0, Right, Top, Bottom };
    typedef int Edge;

public:
    explicit Tausch(int localDimX, int localDimY, int mpiNumX, int mpiNumY);
    ~Tausch();

    void enableOpenCL(bool blockingSyncCpuGpu, bool setupOpenCL = false, int clLocalWorkgroupSize = 64, bool giveOpenCLDeviceName = false);
    void enableOpenCL(cl::Device &cl_defaultDevice, cl::Context &cl_context, cl::CommandQueue &cl_queue, bool blockingSyncCpuGpu, int clLocalWorkgroupSize = 64);

    void postCpuReceives();

    void performCpuToCpu() { startCpuEdge(Left); startCpuEdge(Right); completeCpuEdge(Left); completeCpuEdge(Right);
                             startCpuEdge(Top); startCpuEdge(Bottom); completeCpuEdge(Top); completeCpuEdge(Bottom); }

    void performCpuToGpu() { startCpuToGpu(); completeCpuToGpu(); }

    void performCpuToCpuAndCpuToGpu() { startCpuEdge(Left); startCpuEdge(Right); startCpuToGpu();
                                        completeCpuEdge(Left); completeCpuEdge(Right); startCpuEdge(Top); startCpuEdge(Bottom);
                                        completeCpuToGpu(); completeCpuEdge(Top); completeCpuEdge(Bottom); }

    void performGpuToCpu() { startGpuToCpu(); completeGpuToCpu(); }

    void startCpuToGpu();
    void startGpuToCpu();

    void completeCpuToGpu();
    void completeGpuToCpu();

    void startCpuEdge(Edge edge);
    void completeCpuEdge(Edge edge);

    void syncCpuAndGpu();

    void setMPICommunicator(MPI_Comm comm) { TAUSCH_COMM = comm; }
    MPI_Comm getMPICommunicator() { return TAUSCH_COMM; }


    void setHaloWidth(int haloWidth) { this->haloWidth = 1; }
    void setCPUData(real_t *dat);
    void setGPUData(cl::Buffer &dat, int gpuDimX, int gpuDimY);
    bool isGpuEnabled() { return gpuEnabled; }

    cl::Context cl_context;
    cl::CommandQueue cl_queue;

    void checkOpenCLError(cl_int clErr, std::string loc);

private:
    void EveryoneOutput(const std::string &inMessage);

    int localDimX, localDimY;
    real_t *cpuData;
    cl::Buffer gpuData;

    int haloWidth;
    int gpuDimX, gpuDimY;
    cl::Buffer cl_gpuDimX, cl_gpuDimY;

    int mpiRank, mpiSize;
    int mpiNumX, mpiNumY;

    void setupOpenCL(bool giveOpenCLDeviceName);
    void compileKernels();

    real_t **cpuToCpuSendBuffer;
    real_t **cpuToCpuRecvBuffer;

    real_t *cpuToGpuBuffer;
    real_t *gpuToCpuBuffer;

    cl::Platform cl_platform;
    cl::Device cl_defaultDevice;
    cl::Program cl_programs;

    cl::Buffer cl_gpuToCpuBuffer;
    cl::Buffer cl_cpuToGpuBuffer;

    bool gpuEnabled;
    bool gpuInfoGiven;
    bool cpuInfoGiven;
    bool cpuRecvsPosted;

    bool cpuStarted[4];

    bool cpuToGpuStarted;
    bool gpuToCpuStarted;

    // this refers to inter-partition boundaries
    bool haveBoundary[4];

    int cl_kernelLocalSize;

    MPI_Comm TAUSCH_COMM;

    MPI_Request cpuToCpuSendRequest[4];
    MPI_Request cpuToCpuRecvRequest[4];

    std::atomic<int> sync_counter;
    std::atomic<int> sync_lock;

    bool blockingSyncCpuGpu;

};

#endif
