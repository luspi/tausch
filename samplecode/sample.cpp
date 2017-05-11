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

Sample::Sample(int localDimX, int localDimY, int gpuDimX, int gpuDimY, int loops, int cpuHaloWidth[4], int gpuHaloWidth[4], int mpiNumX, int mpiNumY, bool cpuonly, int clWorkGroupSize, bool giveOpenCLDeviceName) {

    // obtain MPI rank
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    // the overall x and y dimension of the local partition
    dimX = localDimX, dimY = localDimY;
    this->gpuDimX = gpuDimX, this->gpuDimY = gpuDimX;
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
    for(int i = 0; i < 4; ++i)
        this->cpuHaloWidth[i] = cpuHaloWidth[i];
    for(int i = 0; i < 4; ++i)
        this->gpuHaloWidth[i] = gpuHaloWidth[i];

    tausch = new Tausch2D(localDimX, localDimY, mpiNumX, mpiNumY, cpuHaloWidth);
    if(!cpuonly) tausch->enableOpenCL(gpuHaloWidth, true, clWorkGroupSize, giveOpenCLDeviceName);

    // how many points overall in the mesh and a CPU buffer for all of them
    int num = (dimX+cpuHaloWidth[Tausch2D::LEFT]+cpuHaloWidth[Tausch2D::RIGHT])*(dimY+cpuHaloWidth[Tausch2D::TOP]+cpuHaloWidth[Tausch2D::BOTTOM]);
    datCPU = new real_t[num]{};

    stencilNumPoints = 5;
    stencil = new real_t[num*5]{};

    if(cpuonly) {
        for(int j = 0; j < dimY; ++j)
            for(int i = 0; i < dimX; ++i)
                datCPU[(j+cpuHaloWidth[Tausch2D::BOTTOM])*(dimX+cpuHaloWidth[Tausch2D::LEFT]+cpuHaloWidth[Tausch2D::RIGHT]) + i+cpuHaloWidth[Tausch2D::LEFT]] = j*dimX+i+1;

        for(int y = 0; y < dimY; ++y) {
            for(int x = 0; x < dimX; ++x) {
                int index = (y+cpuHaloWidth[Tausch2D::BOTTOM])*(dimX+cpuHaloWidth[Tausch2D::LEFT]+cpuHaloWidth[Tausch2D::RIGHT])+x+cpuHaloWidth[Tausch2D::LEFT];
                stencil[stencilNumPoints*index + 0] = index*10+1;
                stencil[stencilNumPoints*index + 1] = index*10+2;
                stencil[stencilNumPoints*index + 2] = index*10+4;
                stencil[stencilNumPoints*index + 3] = index*10+5;
                stencil[stencilNumPoints*index + 4] = index*10+7;
            }
        }
    } else {
        for(int j = 0; j < dimY; ++j)
            for(int i = 0; i < dimX; ++i)
                if(!(i >= (dimX-gpuDimX)/2 && i < (dimX-gpuDimX)/2+gpuDimX
                     && j >= (dimY-gpuDimY)/2 && j < (dimY-gpuDimY)/2+gpuDimY))
                    datCPU[(j+cpuHaloWidth[Tausch2D::BOTTOM])*(dimX+cpuHaloWidth[Tausch2D::LEFT]+cpuHaloWidth[Tausch2D::RIGHT]) + i+cpuHaloWidth[Tausch2D::LEFT]] = j*dimX+i+1;

        for(int y = 0; y < dimY; ++y) {
            for(int x = 0; x < dimX; ++x) {
                if(!(x >= (dimX-gpuDimX)/2 && x < (dimX-gpuDimX)/2+gpuDimX
                     && y >= (dimY-gpuDimY)/2 && y < (dimY-gpuDimY)/2+gpuDimY)) {
                    int index = (y+cpuHaloWidth[Tausch2D::BOTTOM])*(dimX+cpuHaloWidth[Tausch2D::LEFT]+cpuHaloWidth[Tausch2D::RIGHT])+x+cpuHaloWidth[Tausch2D::LEFT];
                    stencil[stencilNumPoints*index + 0] = index*10+1;
                    stencil[stencilNumPoints*index + 1] = index*10+2;
                    stencil[stencilNumPoints*index + 2] = index*10+4;
                    stencil[stencilNumPoints*index + 3] = index*10+5;
                    stencil[stencilNumPoints*index + 4] = index*10+7;
                }
            }
        }
    }


    if(!cpuonly) {

        // how many points only on the device and an OpenCL buffer for them
        datGPU = new real_t[(gpuDimX+gpuHaloWidth[Tausch2D::LEFT]+gpuHaloWidth[Tausch2D::RIGHT])*(gpuDimY+gpuHaloWidth[Tausch2D::TOP]+gpuHaloWidth[Tausch2D::BOTTOM])]{};
        stencilGPU = new real_t[stencilNumPoints*(gpuDimX+gpuHaloWidth[Tausch2D::LEFT]+gpuHaloWidth[Tausch2D::RIGHT])*(gpuDimY+gpuHaloWidth[Tausch2D::TOP]+gpuHaloWidth[Tausch2D::BOTTOM])]{};

        for(int j = 0; j < gpuDimY; ++j)
            for(int i = 0; i < gpuDimX; ++i)
                datGPU[(j+gpuHaloWidth[Tausch2D::BOTTOM])*(gpuDimX+gpuHaloWidth[Tausch2D::LEFT]+gpuHaloWidth[Tausch2D::RIGHT]) + i+gpuHaloWidth[Tausch2D::LEFT]] = j*gpuDimX+i+1;
        for(int j = 0; j < gpuDimY; ++j)
            for(int i = 0; i < gpuDimX; ++i) {
                int index = (j+gpuHaloWidth[Tausch2D::BOTTOM])*(gpuDimX+gpuHaloWidth[Tausch2D::LEFT]+gpuHaloWidth[Tausch2D::RIGHT]) + i+gpuHaloWidth[Tausch2D::LEFT];
                stencilGPU[stencilNumPoints*index + 0] = index*10+1;
                stencilGPU[stencilNumPoints*index + 1] = index*10+2;
                stencilGPU[stencilNumPoints*index + 2] = index*10+4;
                stencilGPU[stencilNumPoints*index + 3] = index*10+5;
                stencilGPU[stencilNumPoints*index + 4] = index*10+7;
            }

        try {

            cl_datGpu = cl::Buffer(tausch->getContext(), &datGPU[0], (&datGPU[(gpuDimX+gpuHaloWidth[Tausch2D::LEFT]+gpuHaloWidth[Tausch2D::RIGHT])*(gpuDimY+gpuHaloWidth[Tausch2D::TOP]+gpuHaloWidth[Tausch2D::BOTTOM])-1])+1, false);
            cl_stencilGPU = cl::Buffer(tausch->getContext(), &stencilGPU[0], (&stencilGPU[stencilNumPoints*(gpuDimX+gpuHaloWidth[Tausch2D::LEFT]+gpuHaloWidth[Tausch2D::RIGHT])*(gpuDimY+gpuHaloWidth[Tausch2D::TOP]+gpuHaloWidth[Tausch2D::BOTTOM]) -1])+1, false);

        } catch(cl::Error error) {
            std::cout << "[sample] OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
            exit(1);
        }

    } else
        datGPU = new real_t[1]{};

    // pass pointers to the two data containers
    tausch->setCPUData(datCPU);
    tausch->setCPUStencil(stencil, stencilNumPoints);
    if(!cpuonly) {
        tausch->setGPUData(cl_datGpu, gpuDimX, gpuDimY);
        tausch->setGPUStencil(cl_stencilGPU, stencilNumPoints);
    }

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
        } else {

            tausch->postCpuDataReceives();
            tausch->postCpuStencilReceives();

            tausch->startCpuDataEdge(Tausch2D::LEFT); tausch->startCpuDataEdge(Tausch2D::RIGHT);
            tausch->startCpuStencilEdge(Tausch2D::LEFT); tausch->startCpuStencilEdge(Tausch2D::RIGHT);

            tausch->startCpuToGpuData(); tausch->startCpuToGpuStencil();

            tausch->completeCpuDataEdge(Tausch2D::LEFT); tausch->completeCpuDataEdge(Tausch2D::RIGHT); tausch->startCpuDataEdge(Tausch2D::TOP); tausch->startCpuDataEdge(Tausch2D::BOTTOM);
            tausch->completeCpuStencilEdge(Tausch2D::LEFT); tausch->completeCpuStencilEdge(Tausch2D::RIGHT); tausch->startCpuStencilEdge(Tausch2D::TOP); tausch->startCpuStencilEdge(Tausch2D::BOTTOM);

            tausch->completeGpuToCpuData(); tausch->completeGpuToCpuStencil();

            tausch->completeCpuDataEdge(Tausch2D::TOP); tausch->completeCpuDataEdge(Tausch2D::BOTTOM);
            tausch->completeCpuStencilEdge(Tausch2D::TOP); tausch->completeCpuStencilEdge(Tausch2D::BOTTOM);

        }

    }

}

void Sample::launchGPU() {

    if(cpuonly) return;

    for(int run = 0; run < loops; ++run) {
        tausch->startGpuToCpuData(); tausch->startGpuToCpuStencil();
        tausch->completeCpuToGpuData(); tausch->completeCpuToGpuStencil();
    }

    try {
        cl::copy(tausch->getQueue(), cl_datGpu, &datGPU[0], (&datGPU[(gpuDimX+gpuHaloWidth[Tausch2D::LEFT]+gpuHaloWidth[Tausch2D::RIGHT])*(gpuDimY+gpuHaloWidth[Tausch2D::TOP]+gpuHaloWidth[Tausch2D::BOTTOM])-1])+1);
        cl::copy(tausch->getQueue(), cl_stencilGPU, &stencilGPU[0], (&stencilGPU[stencilNumPoints*((gpuDimX+gpuHaloWidth[Tausch2D::LEFT]+gpuHaloWidth[Tausch2D::RIGHT])*(gpuDimY+gpuHaloWidth[Tausch2D::TOP]+gpuHaloWidth[Tausch2D::BOTTOM]))-1])+1);
    } catch(cl::Error error) {
        std::cout << "[launchGPU] OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

void Sample::printCPU() {

    if(cpuonly) {

        for(int j = dimY+cpuHaloWidth[Tausch2D::TOP]+cpuHaloWidth[Tausch2D::BOTTOM]-1; j >= 0; --j) {
            for(int i = 0; i < dimX+cpuHaloWidth[Tausch2D::LEFT]+cpuHaloWidth[Tausch2D::RIGHT]; ++i)
                std::cout << std::setw(3) << datCPU[j*(dimX+cpuHaloWidth[Tausch2D::LEFT]+cpuHaloWidth[Tausch2D::RIGHT]) + i] << " ";
            std::cout << std::endl;
        }

    } else {

        for(int j = dimY+cpuHaloWidth[Tausch2D::TOP]+cpuHaloWidth[Tausch2D::BOTTOM]-1; j >= 0; --j) {
            for(int i = 0; i < dimX+cpuHaloWidth[Tausch2D::LEFT]+cpuHaloWidth[Tausch2D::RIGHT]; ++i) {
                if(i >= (dimX-gpuDimX)/2+cpuHaloWidth[Tausch2D::LEFT]+gpuHaloWidth[Tausch2D::LEFT] && i < (dimX-gpuDimX)/2+cpuHaloWidth[Tausch2D::LEFT]+gpuDimX-gpuHaloWidth[Tausch2D::RIGHT]
                   && j >= (dimY-gpuDimY)/2+cpuHaloWidth[Tausch2D::BOTTOM]+gpuHaloWidth[Tausch2D::BOTTOM] && j < (dimY-gpuDimY)/2+cpuHaloWidth[Tausch2D::BOTTOM]+gpuDimY-gpuHaloWidth[Tausch2D::TOP])
                    std::cout << std::setw(3) << "    ";
                else
                    std::cout << std::setw(3) << datCPU[j*(dimX+cpuHaloWidth[Tausch2D::LEFT]+cpuHaloWidth[Tausch2D::RIGHT]) + i] << " ";
            }
            std::cout << std::endl;
        }

    }

}

void Sample::printCPUStencil() {

    if(cpuonly) {

        for(int j = 0; j < dimY+cpuHaloWidth[Tausch2D::TOP]+cpuHaloWidth[Tausch2D::BOTTOM]; ++j) {
            for(int i = 0; i < dimX+cpuHaloWidth[Tausch2D::LEFT]+cpuHaloWidth[Tausch2D::RIGHT]; ++i) {
                std::cout << j*(dimX+cpuHaloWidth[Tausch2D::LEFT]+cpuHaloWidth[Tausch2D::RIGHT]) + i << " :: ";
                for(int s = 0; s < stencilNumPoints; ++s)
                          std::cout << stencil[stencilNumPoints*(j*(dimX+cpuHaloWidth[Tausch2D::LEFT]+cpuHaloWidth[Tausch2D::RIGHT]) + i) + s] << (s != stencilNumPoints-1 ? "/" : "");
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }

    } else {

        for(int j = 0; j < dimY+cpuHaloWidth[Tausch2D::TOP]+cpuHaloWidth[Tausch2D::BOTTOM]; ++j) {
            for(int i = 0; i < dimX+cpuHaloWidth[Tausch2D::LEFT]+cpuHaloWidth[Tausch2D::RIGHT]; ++i) {
                std::cout << j*(dimX+cpuHaloWidth[Tausch2D::LEFT]+cpuHaloWidth[Tausch2D::RIGHT]) + i << " :: ";
                if(i >= (dimX-gpuDimX)/2+cpuHaloWidth[Tausch2D::LEFT]+gpuHaloWidth[Tausch2D::LEFT] && i < (dimX-gpuDimX)/2+cpuHaloWidth[Tausch2D::LEFT]+gpuDimX-gpuHaloWidth[Tausch2D::RIGHT]
                   && j >= (dimY-gpuDimY)/2+cpuHaloWidth[Tausch2D::BOTTOM]+gpuHaloWidth[Tausch2D::BOTTOM] && j < (dimY-gpuDimY)/2+cpuHaloWidth[Tausch2D::BOTTOM]+gpuDimY-gpuHaloWidth[Tausch2D::TOP]) {
                    std::cout << "---------" << std::endl;
                    continue;
                }
                for(int s = 0; s < stencilNumPoints; ++s)
                          std::cout << stencil[stencilNumPoints*(j*(dimX+cpuHaloWidth[Tausch2D::LEFT]+cpuHaloWidth[Tausch2D::RIGHT]) + i) + s] << (s != stencilNumPoints-1 ? "/" : "");
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }

    }

}

void Sample::printGPU() {

    for(int i = gpuDimY+gpuHaloWidth[Tausch2D::TOP]+gpuHaloWidth[Tausch2D::BOTTOM] -1; i >= 0; --i) {
        for(int j = 0; j < gpuDimX+gpuHaloWidth[Tausch2D::LEFT]+gpuHaloWidth[Tausch2D::RIGHT]; ++j)
            std::cout << std::setw(3) << datGPU[i*(gpuDimX+gpuHaloWidth[Tausch2D::LEFT]+gpuHaloWidth[Tausch2D::RIGHT]) + j] << " ";
        std::cout << std::endl;
    }

}

void Sample::printGPUStencil() {

    for(int j = 0; j < gpuDimY+gpuHaloWidth[Tausch2D::TOP]+gpuHaloWidth[Tausch2D::BOTTOM]; ++j) {
        for(int i = 0; i < gpuDimX+gpuHaloWidth[Tausch2D::LEFT]+gpuHaloWidth[Tausch2D::RIGHT]; ++i) {
            std::cout << j*(gpuDimX+gpuHaloWidth[Tausch2D::LEFT]+gpuHaloWidth[Tausch2D::RIGHT]) + i << " :: ";
            for(int s = 0; s < stencilNumPoints; ++s)
                      std::cout << stencilGPU[stencilNumPoints*(j*(gpuDimX+gpuHaloWidth[Tausch2D::LEFT]+gpuHaloWidth[Tausch2D::RIGHT]) + i) + s] << (s != stencilNumPoints-1 ? "/" : "");
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

}

