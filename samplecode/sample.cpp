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

Sample::Sample(int localDimX, int localDimY, real_t portionGPU, int loops, int cpuHaloWidth, int gpuHaloWidth, int mpiNumX, int mpiNumY, bool cpuonly, int clWorkGroupSize, bool giveOpenCLDeviceName) {

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

    // the width of the halos
    this->cpuHaloWidth = cpuHaloWidth;
    this->gpuHaloWidth = gpuHaloWidth;

    tausch = new Tausch2D(localDimX, localDimY, mpiNumX, mpiNumY, cpuHaloWidth);
    if(!cpuonly) tausch->enableOpenCL(gpuHaloWidth, true, clWorkGroupSize, giveOpenCLDeviceName);

    // how many points overall in the mesh and a CPU buffer for all of them
    int num = (dimX+2*cpuHaloWidth)*(dimY+2*cpuHaloWidth);
    datCPU = new real_t[num]{};

    if(cpuonly) {
        for(int j = 0; j < dimY; ++j)
            for(int i = 0; i < dimX; ++i)
                datCPU[(j+cpuHaloWidth)*(dimX+2*cpuHaloWidth) + i+cpuHaloWidth] = j*dimX+i+1;
    } else {
        for(int j = 0; j < dimY; ++j)
            for(int i = 0; i < dimX; ++i)
                if(!(i >= (dimX-gpuDimX)/2 && i < (dimX-gpuDimX)/2+gpuDimX
                     && j >= (dimY-gpuDimY)/2 && j < (dimY-gpuDimY)/2+gpuDimY))
                    datCPU[(j+cpuHaloWidth)*(dimX+2*cpuHaloWidth) + i+cpuHaloWidth] = j*dimX+i+1;
    }

    stencilNumPoints = 5;
    stencil = new real_t[num*5];
    for(int y = 0; y < dimY; ++y) {
        for(int x = 0; x < dimX; ++x) {
            stencil[stencilNumPoints*((y+cpuHaloWidth)*(dimX+2*cpuHaloWidth)+x+cpuHaloWidth) + 0] = 1;
            stencil[stencilNumPoints*((y+cpuHaloWidth)*(dimX+2*cpuHaloWidth)+x+cpuHaloWidth) + 1] = 2;
            stencil[stencilNumPoints*((y+cpuHaloWidth)*(dimX+2*cpuHaloWidth)+x+cpuHaloWidth) + 2] = 4;
            stencil[stencilNumPoints*((y+cpuHaloWidth)*(dimX+2*cpuHaloWidth)+x+cpuHaloWidth) + 3] = 5;
            stencil[stencilNumPoints*((y+cpuHaloWidth)*(dimX+2*cpuHaloWidth)+x+cpuHaloWidth) + 4] = 7;
        }
    }


    if(!cpuonly) {

        // how many points only on the device and an OpenCL buffer for them
        datGPU = new real_t[(gpuDimX+2*gpuHaloWidth)*(gpuDimY+2*gpuHaloWidth)]{};

        for(int j = 0; j < gpuDimY; ++j)
            for(int i = 0; i < gpuDimX; ++i)
                datGPU[(j+gpuHaloWidth)*(gpuDimX+2*gpuHaloWidth) + i+gpuHaloWidth] = j*gpuDimX+i+1;

        try {

            cl_datGpu = cl::Buffer(tausch->getContext(), &datGPU[0], (&datGPU[(gpuDimX+2*gpuHaloWidth)*(gpuDimY+2*gpuHaloWidth)-1])+1, false);

        } catch(cl::Error error) {
            std::cout << "[sample] OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
            exit(1);
        }

    } else
        datGPU = new real_t[1]{};

    // pass pointers to the two data containers
    tausch->setCPUData(datCPU);
    tausch->setCPUStencil(stencil, stencilNumPoints);
    if(!cpuonly)
        tausch->setGPUData(cl_datGpu, gpuDimX, gpuDimY);

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

        if(cpuonly) {
            tausch->performCpuToCpuData();
            tausch->performCpuToCpuStencil();
        } else
            tausch->performCpuToCpuDataAndCpuToGpuData();

    }

}

void Sample::launchGPU() {

    if(cpuonly) return;

    for(int run = 0; run < loops; ++run)
        tausch->performGpuToCpu();

    try {
        cl::copy(tausch->getQueue(), cl_datGpu, &datGPU[0], (&datGPU[(gpuDimX+2*gpuHaloWidth)*(gpuDimY+2*gpuHaloWidth)-1])+1);
    } catch(cl::Error error) {
        std::cout << "[launchGPU] OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

void Sample::printCPU() {

    if(cpuonly) {

        for(int j = dimY+2*cpuHaloWidth-1; j >= 0; --j) {
            for(int i = 0; i < dimX+2*cpuHaloWidth; ++i)
                std::cout << std::setw(3) << datCPU[j*(dimX+2*cpuHaloWidth) + i] << " ";
            std::cout << std::endl;
        }

    } else {

        for(int j = dimY+2*cpuHaloWidth-1; j >= 0; --j) {
            for(int i = 0; i < dimX+2*cpuHaloWidth; ++i) {
                if(i-cpuHaloWidth-gpuHaloWidth >= (dimX-gpuDimX)/2 && i-(cpuHaloWidth-gpuHaloWidth) < (dimX-gpuDimX)/2+gpuDimX
                     && j-cpuHaloWidth-gpuHaloWidth >= (dimY-gpuDimY)/2 && j-(cpuHaloWidth-gpuHaloWidth) < (dimY-gpuDimY)/2+gpuDimY)
                    std::cout << std::setw(3) << "    ";
                else
                    std::cout << std::setw(3) << datCPU[j*(dimX+2*cpuHaloWidth) + i] << " ";
            }
            std::cout << std::endl;
        }

    }

}

void Sample::printCPUStencil() {

    if(cpuonly) {

        for(int j = 0; j < dimY+2*cpuHaloWidth; ++j) {
            for(int i = 0; i < dimX+2*cpuHaloWidth; ++i) {
                std::cout << j*(dimX+2*cpuHaloWidth) + i << " :: ";
                for(int s = 0; s < stencilNumPoints; ++s)
                          std::cout << stencil[stencilNumPoints*(j*(dimX+2*cpuHaloWidth) + i) + s] << (s != stencilNumPoints-1 ? "/" : "");
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }

    } else {

    }

}

void Sample::printGPU() {

    for(int i = gpuDimY+2*gpuHaloWidth -1; i >= 0; --i) {
        for(int j = 0; j < gpuDimX+2*gpuHaloWidth; ++j)
            std::cout << std::setw(3) << datGPU[i*(gpuDimX+2*gpuHaloWidth) + j] << " ";
        std::cout << std::endl;
    }

}

