#ifndef _TAUSCH_H
#define _TAUSCH_H

#include <vector>
#include <mpi.h>
#include <fstream>
#include <cmath>
#include <sstream>
#include <iostream>
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

class Tausch {

    enum {
        Left = 0,
        Right,
        Top,
        Bottom
    };
    typedef int Edge;

public:
    explicit Tausch(int localDimX, int localDimY, int mpiNumX, int mpiNumY, bool withOpenCL = false, bool setupOpenCL = false);
    ~Tausch();

    void setOpenCLInfo(cl::Device &cl_defaultDevice, cl::Context &cl_context, cl::CommandQueue &cl_queue);

    void postCpuReceives();
    void postGpuReceives();

    void startCpuTausch();
    void completeCpuTausch();
    void startAndCompleteCpuTausch() { startCpuTausch(); completeCpuTausch(); }
    void startGpuTausch();
    void completeGpuTausch();
    void startAndCompleteGpuTausch() { startGpuTausch(); completeGpuTausch(); }

    void syncCpuWaitsForGpu(bool iAmTheCPU);
    void syncGpuWaitsForCpu(bool iAmTheCPU);
    void syncCpuAndGpu(bool iAmTheCPU);

    cl::Context cl_context;
    cl::CommandQueue cl_queue;

    void setHaloWidth(int haloWidth) { this->haloWidth = 1; }
    void setCPUData(double *dat);
    void setGPUData(cl::Buffer &dat, int gpuWidth, int gpuHeight);
    bool isGpuEnabled() { return gpuEnabled; }

private:
    int localDimX, localDimY;
    double *cpuData;
    cl::Buffer gpuData;

    int haloWidth;
    int gpuWidth, gpuHeight;
    cl::Buffer cl_gpuWidth, cl_gpuHeight;

    int mpiRank, mpiSize;
    int mpiNumX, mpiNumY;

    void setupOpenCL();
    void compileKernels();

    double *cpuToCpuSendBuffer;
    double *cpuToCpuRecvBuffer;
    double *cpuToGpuSendBuffer;
    double *cpuToGpuRecvBuffer;
    double *gpuToCpuSendBuffer;
    double *gpuToCpuRecvBuffer;

    cl::Platform cl_platform;
    cl::Device cl_defaultDevice;
    cl::Program cl_programs;

    cl::Buffer cl_gpuToCpuSendBuffer;
    cl::Buffer cl_gpuToCpuRecvBuffer;

    bool gpuEnabled;
    bool gpuInfoGiven;
    bool cpuInfoGiven;
    bool cpuRecvsPosted;
    bool gpuRecvsPosted;
    bool cpuStarted;
    bool gpuStarted;

    // This refers to the overall domain border
    bool haveLeftBorder;
    bool haveRightBorder;
    bool haveTopBorder;
    bool haveBottomBorder;

    // this refers to inter-partition boundaries
    bool haveLeftBoundary;
    bool haveRightBoundary;
    bool haveTopBoundary;
    bool haveBottomBoundary;

    std::vector<MPI_Request> allCpuRequests;
    std::vector<MPI_Request> allGpuRequests;

    int cl_kernelLocalSize;



    MPI_Request cpuToCpuLeftSendRequest;
    MPI_Request cpuToCpuRightSendRequest;
    MPI_Request cpuToCpuTopSendRequest;
    MPI_Request cpuToCpuBottomSendRequest;
    MPI_Request cpuToCpuLeftRecvRequest;
    MPI_Request cpuToCpuRightRecvRequest;
    MPI_Request cpuToCpuTopRecvRequest;
    MPI_Request cpuToCpuBottomRecvRequest;

    MPI_Request cpuToCpuBottomLeftSendRequest;
    MPI_Request cpuToCpuBottomRightSendRequest;
    MPI_Request cpuToCpuTopLeftSendRequest;
    MPI_Request cpuToCpuTopRightSendRequest;
    MPI_Request cpuToCpuBottomLeftRecvRequest;
    MPI_Request cpuToCpuBottomRightRecvRequest;
    MPI_Request cpuToCpuTopLeftRecvRequest;
    MPI_Request cpuToCpuTopRightRecvRequest;

    MPI_Request cpuToGpuLeftSendRequest;
    MPI_Request cpuToGpuRightSendRequest;
    MPI_Request cpuToGpuTopSendRequest;
    MPI_Request cpuToGpuBottomSendRequest;
    MPI_Request cpuToGpuLeftRecvRequest;
    MPI_Request cpuToGpuRightRecvRequest;
    MPI_Request cpuToGpuTopRecvRequest;
    MPI_Request cpuToGpuBottomRecvRequest;

    MPI_Request gpuToCpuLeftSendRequest;
    MPI_Request gpuToCpuRightSendRequest;
    MPI_Request gpuToCpuTopSendRequest;
    MPI_Request gpuToCpuBottomSendRequest;
    MPI_Request gpuToCpuLeftRecvRequest;
    MPI_Request gpuToCpuRightRecvRequest;
    MPI_Request gpuToCpuTopRecvRequest;
    MPI_Request gpuToCpuBottomRecvRequest;


};

#endif
