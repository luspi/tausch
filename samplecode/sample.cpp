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

Sample::Sample(int localDimX, int localDimY, double portionGPU, int loops, int mpiNumX, int mpiNumY, bool cpuonly, int clWorkGroupSize, bool giveOpenCLDeviceName) {

    // obtain MPI rank
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    // the overall x and y dimension of the local partition
    dimX = localDimX, dimY = localDimY;
    gpuDimX = dimX*std::sqrt(portionGPU), gpuDimY = dimY*std::sqrt(portionGPU);
    this->loops = loops;
    this->cpuonly = cpuonly;

    if(mpiNumX == 0 || mpiNumY == 0) {
        mpiNumX = std::sqrt(mpiSize);
        mpiNumY = mpiNumX;
    }

    if(mpiNumX*mpiNumY != mpiSize) {
        std::cout << "ERROR: Total number of MPI ranks requested (" << mpiNumX << "x" << mpiNumY << ") doesn't match global MPI size of " << mpiSize << "... Abort!" << std::endl;
        exit(1);
    }

    tau = new Tausch(localDimX, localDimY, mpiNumX, mpiNumY, !cpuonly, true, clWorkGroupSize, giveOpenCLDeviceName);

    // the width of the halos
    int halowidth = 1;

    // how many points overall in the mesh and a CPU buffer for all of them
    int num = (dimX+2)*(dimY+2);
    datCPU = new double[num]{};

    for(int j = 0; j < dimY; ++j)
        for(int i = 0; i < dimX; ++i)
            if(!(i >= (dimX-gpuDimX)/2 && i < (dimX-gpuDimX)/2+gpuDimX
               && j >= (dimY-gpuDimY)/2 && j < (dimY-gpuDimY)/2+gpuDimY))
                datCPU[(j+1)*(dimX+2) + i+1] = j*dimX+i;


    // how many points only on the device and an OpenCL buffer for them
    datGPU = new double[(gpuDimX+2)*(gpuDimY+2)]{};

    for(int j = 0; j < gpuDimY; ++j)
        for(int i = 0; i < gpuDimX; ++i)
            datGPU[(j+1)*(gpuDimX+2) + i+1] = j*gpuDimX+i;


    if(!cpuonly) {

        try {

            cl_datGpu = cl::Buffer(tau->cl_context, &datGPU[0], (&datGPU[(gpuDimX+2)*(gpuDimY+2)-1])+1, false);

        } catch(cl::Error error) {
            std::cout << "[sample] OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
            exit(1);
        }

    }

    // currently redundant, can only be 1
    tau->setHaloWidth(1);

    // pass pointers to the two data containers
    tau->setCPUData(datCPU);
    if(!cpuonly)
        tau->setGPUData(cl_datGpu, gpuDimX, gpuDimY);

}

Sample::~Sample() {

    delete[] datCPU;
    delete[] datGPU;

}

void Sample::launchCPU() {

    for(int run = 0; run < loops; ++run) {

        // post the receives
        tau->postCpuReceives();

        tau->performCpuToCpuTausch();

        if(!cpuonly) {

            tau->startCpuToGpuTausch();
            tau->completeCpuToGpuTausch();

        }

    }

}

void Sample::launchGPU() {

    if(cpuonly) return;

    for(int run = 0; run < loops; ++run) {

        tau->startGpuToCpuTausch();

        tau->completeGpuToCpuTausch();

    }

    try {
        cl::copy(tau->cl_queue, cl_datGpu, &datGPU[0], (&datGPU[(gpuDimX+2)*(gpuDimY+2)-1])+1);
    } catch(cl::Error error) {
        std::cout << "[launchGPU] OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

void Sample::printCPU() {

    std::stringstream ss;

    for(int j = 0; j < dimY+2; ++j) {
        std:: stringstream tmp;
        for(int i = 0; i < dimX+2; ++i) {
            if(i-1 > (dimX-gpuDimX)/2 && i < (dimX-gpuDimX)/2+gpuDimX
               && j-1 > (dimY-gpuDimY)/2 && j < (dimY-gpuDimY)/2+gpuDimY)
                tmp << std::setw(3) << "    ";
            else
                tmp << std::setw(3) << datCPU[j*(dimX+2) + i] << " ";
        }
        tmp << std::endl;
        std::string str = ss.str();
        ss.clear();
        ss.str("");
        ss << tmp.str() << str;
    }

    std::cout << ss.str();

}

void Sample::printGPU() {

    std::stringstream ss;

    for(int i = 0; i < gpuDimY+2; ++i) {
        std:: stringstream tmp;
        for(int j = 0; j < gpuDimX+2; ++j)
            tmp << std::setw(3) << datGPU[i*(gpuDimX+2) + j] << " ";
        tmp << std::endl;
        std::string str = ss.str();
        ss.clear();
        ss.str("");
        ss << tmp.str() << str;
    }

    std::cout << ss.str();

}

