#include "sample.h"

/*****************************
 *
 * This code uses the following test setup
 *
 *     |             20              |
 *   -----------------------------------
 *     |   CPU       5               |
 *     |                             |
 *     |       ---------------       |
 *  2  |   5   |    GPU      |   5   |
 *  0  |       |             |       |
 *     |       ---------------       |
 *     |                             |
 *     |             5               |
 *   -----------------------------------
 *     |                             |
 *
 *  with a halo width of 1 (for any halo).
 *
 *******************************/

Sample::Sample(int localDimX, int localDimY, double portionGPU, int loops, int mpiNumX, int mpiNumY, bool cpuonly) {

    // obtain MPI rank
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    // the overall x and y dimension of the local partition
    dimX = localDimX, dimY = localDimY;
    gpuDimX = dimX*portionGPU, gpuDimY = dimY*portionGPU;

    if(mpiNumX == 0 || mpiNumY == 0) {
        mpiNumX = std::sqrt(mpiSize);
        mpiNumY = mpiNumX;
    }

    if(mpiNumX*mpiNumY != mpiSize) {
        std::cout << "ERROR: Total number of MPI ranks requested (" << mpiNumX << "x" << mpiNumY << ") doesn't match global MPI size of " << mpiSize << "... Abort!" << std::endl;
        exit(1);
    }

    Tausch tau(dimX, dimY, std::sqrt(mpiSize), std::sqrt(mpiSize), !cpuonly, true);

    // the width of the halos
    int halowidth = 1;

    // how many points overall in the mesh and a CPU buffer for all of them
    int num = (dimX+2)*(dimY+2);
    double *dat = new double[num]{};

    for(int j = 0; j < dimY; ++j)
        for(int i = 0; i < dimX; ++i)
            if(!(i >= (dimX-gpuDimX)/2 && i < (dimX-gpuDimX)/2+gpuDimX
               && j >= (dimY-gpuDimY)/2 && j < (dimY-gpuDimY)/2+gpuDimY))
                dat[(j+1)*(dimX+2) + i+1] = j*dimX+i;


    // how many points only on the device and an OpenCL buffer for them
    int gpunum = (gpuDimX+2)*(gpuDimY+2);
    double *gpudat__host = new double[gpunum]{};

    for(int j = 0; j < gpuDimY; ++j)
        for(int i = 0; i < gpuDimX; ++i)
            gpudat__host[(j+1)*(gpuDimX+2) + i+1] = j*gpuDimX+i;


    cl::Buffer bufdat;

    if(!cpuonly) {

        try {

            bufdat = cl::Buffer(tau.cl_context, &gpudat__host[0], (&gpudat__host[gpunum-1])+1, false);

        } catch(cl::Error error) {
            std::cout << "[sample] OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
            exit(1);
        }

    }

    // currently redundant, can only be 1
    tau.setHaloWidth(1);

    MPI_Barrier(MPI_COMM_WORLD);
    auto tStart = std::chrono::steady_clock::now();

    // pass pointers to the two data containers
    tau.setCPUData(dat);
    if(!cpuonly)
        tau.setGPUData(bufdat, gpuDimX, gpuDimY);

    for(int run = 0; run < loops; ++run) {

        // post the receives
        tau.postCpuReceives();
        if(!cpuonly)
            tau.postGpuReceives();

        // initiate sending of the data
        tau.startCpuTausch();
        if(!cpuonly)
            tau.startGpuTausch();

        // Wait for communication to be finished and distribute result
        tau.completeCpuTausch();
        if(!cpuonly)
            tau.completeGpuTausch();

    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto tEnd = std::chrono::steady_clock::now();

    double totalTime = std::chrono::duration<double, std::milli>(tEnd-tStart).count();

    if(mpiRank == 0)
        std::cout << " Total time required: " << totalTime << " ms" << std::endl;

    delete[] dat;
    delete[] gpudat__host;

}

void Sample::EveryoneOutput(const std::string &inMessage) {

    for(int iRank = 0; iRank < mpiSize; ++iRank){
        if(mpiRank == iRank)
            std::cout << inMessage;
        MPI_Barrier(MPI_COMM_WORLD);
    }

}

void Sample::printCPU(double *dat) {

    std::stringstream ss;

    for(int j = 0; j < dimY+2; ++j) {
        std:: stringstream tmp;
        for(int i = 0; i < dimX+2; ++i) {
            if(i-1 > (dimX-gpuDimX)/2 && i < (dimX-gpuDimX)/2+gpuDimX
               && j-1 > (dimY-gpuDimY)/2 && j < (dimY-gpuDimY)/2+gpuDimY)
                tmp << std::setw(3) << "    ";
            else
                tmp << std::setw(3) << dat[j*(dimX+2) + i] << " ";
        }
        tmp << std::endl;
        std::string str = ss.str();
        ss.clear();
        ss.str("");
        ss << tmp.str() << str;
    }

    std::cout << ss.str();

}

void Sample::printGPU(double *dat) {

    std::stringstream ss;

    for(int i = 0; i < gpuDimY+2; ++i) {
        std:: stringstream tmp;
        for(int j = 0; j < gpuDimX+2; ++j)
            tmp << std::setw(3) << dat[i*(gpuDimX+2) + j] << " ";
        tmp << std::endl;
        std::string str = ss.str();
        ss.clear();
        ss.str("");
        ss << tmp.str() << str;
    }

    std::cout << ss.str();

}

