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

Sample::Sample(int localDimX, int localDimY, int localDimZ, real_t portionGPU, int loops, int haloWidth, int mpiNumX, int mpiNumY, int mpiNumZ, bool cpuonly, int clWorkGroupSize, bool giveOpenCLDeviceName) {

    // obtain MPI rank
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    // the overall x and y dimension of the local partition
    dimX = localDimX, dimY = localDimY, dimZ = localDimZ;
    gpuDimX = dimX*std::cbrt(portionGPU), gpuDimY = dimY*std::cbrt(portionGPU), gpuDimZ = dimZ*std::cbrt(portionGPU);
    this->loops = loops;
    this->cpuonly = cpuonly;

    if(mpiNumX == 0 || mpiNumY == 0 || mpiNumZ == 0) {
        mpiNumX = std::sqrt(mpiSize);
        mpiNumY = mpiNumX;
        mpiNumZ = 1;
    }

    if(mpiNumX*mpiNumY*mpiNumZ != mpiSize) {
        std::cout << "ERROR: Total number of MPI ranks requested (" << mpiNumX << "x" << mpiNumY << "x" << mpiNumZ << ") doesn't match global MPI size of " << mpiSize << "... Abort!" << std::endl;
        exit(1);
    }

    // the width of the halos
    this->haloWidth = haloWidth;

    tausch = new Tausch3D(localDimX, localDimY, localDimZ, mpiNumX, mpiNumY, mpiNumZ, haloWidth);
    if(!cpuonly) tausch->enableOpenCL(true, clWorkGroupSize, giveOpenCLDeviceName);

    // how many points overall in the mesh and a CPU buffer for all of them
    int num = (dimX+2*haloWidth)*(dimY+2*haloWidth)*(dimZ+2*haloWidth);
    datCPU = new real_t[num]{};

    if(cpuonly) {
        for(int z = 0; z < dimZ; ++z)
            for(int j = 0; j < dimY; ++j)
                for(int i = 0; i < dimX; ++i)
                    datCPU[(z+haloWidth)*(dimX+2*haloWidth)*(dimY+2*haloWidth) + (j+haloWidth)*(dimX+2*haloWidth) + i+haloWidth] = z*dimX*dimY+j*dimX+i+1;
    } else {
        for(int j = 0; j < dimY; ++j)
            for(int i = 0; i < dimX; ++i)
                if(!(i >= (dimX-gpuDimX)/2 && i < (dimX-gpuDimX)/2+gpuDimX
                     && j >= (dimY-gpuDimY)/2 && j < (dimY-gpuDimY)/2+gpuDimY))
                    datCPU[(j+haloWidth)*(dimX+2*haloWidth) + i+haloWidth] = j*dimX+i+1;
    }


    if(!cpuonly) {

        // how many points only on the device and an OpenCL buffer for them
        datGPU = new real_t[(gpuDimX+2*haloWidth)*(gpuDimY+2*haloWidth)]{};

        for(int j = 0; j < gpuDimY; ++j)
            for(int i = 0; i < gpuDimX; ++i)
                datGPU[(j+haloWidth)*(gpuDimX+2*haloWidth) + i+haloWidth] = j*gpuDimX+i+1;

        try {

            cl_datGpu = cl::Buffer(tausch->getContext(), &datGPU[0], (&datGPU[(gpuDimX+2*haloWidth)*(gpuDimY+2*haloWidth)-1])+1, false, true);

        } catch(cl::Error error) {
            std::cout << "[sample] OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
            exit(1);
        }

    } else
        datGPU = new real_t[1]{};

    // pass pointers to the two data containers
    tausch->setCPUData(datCPU);
    if(!cpuonly)
        tausch->setGPUData(cl_datGpu, gpuDimX, gpuDimY, gpuDimZ);

}

Sample::~Sample() {

    delete tausch;
    delete[] datCPU;
    delete[] datGPU;

}

void Sample::launchCPU() {

    for(int run = 0; run < loops; ++run) {

        if(mpiRank == 0 && (run+1)%10 == 0)
            std::cout << "Loop " << run+1 << "/" << loops << std::endl;

        // post the receives
        tausch->postCpuReceives();

        if(cpuonly)
            tausch->performCpuToCpu();
        else
            tausch->performCpuToCpuAndCpuToGpu();

    }

}

void Sample::launchGPU() {

    if(cpuonly) return;

    for(int run = 0; run < loops; ++run)
        tausch->performGpuToCpu();

    try {
        cl::copy(tausch->getQueue(), cl_datGpu, &datGPU[0], (&datGPU[(gpuDimX+2)*(gpuDimY+2)-1])+1);
    } catch(cl::Error error) {
        std::cout << "[launchGPU] OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

void Sample::printCPU() {

    if(cpuonly) {

        for(int z = 0; z < dimZ+2*haloWidth; ++z) {

            std::cout << std::endl << "z = " << z << std::endl;

            for(int j = dimY+2*haloWidth-1; j >= 0; --j) {
                for(int i = 0; i < dimX+2*haloWidth; ++i)
                    std::cout << std::setw(3) << datCPU[z*(dimX+2*haloWidth)*(dimY+2*haloWidth) + j*(dimX+2*haloWidth) + i] << " ";
                std::cout << std::endl;
            }

        }

    } else {

        for(int j = dimY+2*haloWidth-1; j >= 0; --j) {
            for(int i = 0; i < dimX+2*haloWidth; ++i) {
                if(i-2*haloWidth >= (dimX-gpuDimX)/2 && i < (dimX-gpuDimX)/2+gpuDimX
                     && j-2*haloWidth >= (dimY-gpuDimY)/2 && j < (dimY-gpuDimY)/2+gpuDimY)
                    std::cout << std::setw(3) << "    ";
                else
                    std::cout << std::setw(3) << datCPU[j*(dimX+2*haloWidth) + i] << " ";
            }
            std::cout << std::endl;
        }

    }

}

void Sample::printGPU() {

    for(int i = gpuDimY+2*haloWidth -1; i >= 0; --i) {
        for(int j = 0; j < gpuDimX+2*haloWidth; ++j)
            std::cout << std::setw(3) << datGPU[i*(gpuDimX+2*haloWidth) + j] << " ";
        std::cout << std::endl;
    }

}

