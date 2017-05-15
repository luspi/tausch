#include "sample.h"

Sample::Sample(int *localDim, int *gpuDim, int loops, int *cpuHaloWidth, int *gpuHaloWidth, int *mpiNum, bool cpuonly, int clWorkGroupSize, bool giveOpenCLDeviceName) {

    // obtain MPI rank
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    // the overall x and y dimension of the local partition
    dimX = localDim[0], dimY = localDim[1], dimZ = localDim[2];
    this->gpuDimX = gpuDim[0], this->gpuDimY = gpuDim[1], this->gpuDimZ = gpuDim[2];
    this->loops = loops;
    this->cpuonly = cpuonly;

    if(mpiNum[0] == 0 || mpiNum[1] == 0 || mpiNum[2] == 0) {
        mpiNum[0] = std::cbrt(mpiSize);
        mpiNum[1] = mpiNum[0];
        mpiNum[2] = mpiNum[0];
    }

    if(mpiNum[0]*mpiNum[1]*mpiNum[2] != mpiSize) {
        std::cout << "ERROR: Total number of MPI ranks requested (" << mpiNum[0] << "x" << mpiNum[1] << "x" << mpiNum[2] << ") doesn't match global MPI size of " << mpiSize << "... Abort!" << std::endl;
        exit(1);
    }

    // the width of the halos
    for(int i = 0; i < 6; ++i)
        this->cpuHaloWidth[i] = cpuHaloWidth[i];
    for(int i = 0; i < 6; ++i)
        this->gpuHaloWidth[i] = gpuHaloWidth[i];

    tausch = new Tausch3D(localDim, mpiNum, cpuHaloWidth);
    if(!cpuonly) tausch->enableOpenCL(gpuHaloWidth, true, clWorkGroupSize, giveOpenCLDeviceName);

    // how many points overall in the mesh and a CPU buffer for all of them
    int num = (dimX+cpuHaloWidth[0]+cpuHaloWidth[1])*(dimY+cpuHaloWidth[2]+cpuHaloWidth[3])*(dimZ+cpuHaloWidth[4]+cpuHaloWidth[5]);
    datCPU = new real_t[num]{};

    if(cpuonly) {
        for(int z = 0; z < dimZ; ++z)
            for(int j = 0; j < dimY; ++j)
                for(int i = 0; i < dimX; ++i)
                    datCPU[(z+cpuHaloWidth[4])*(dimX+cpuHaloWidth[0]+cpuHaloWidth[1])*(dimY+cpuHaloWidth[2]+cpuHaloWidth[3]) + (j+cpuHaloWidth[3])*(dimX+cpuHaloWidth[0]+cpuHaloWidth[1]) + i+cpuHaloWidth[0]] = z*dimX*dimY+j*dimX+i+1;
    } else {
        for(int z = 0; z < dimZ; ++z)
            for(int j = 0; j < dimY; ++j)
                for(int i = 0; i < dimX; ++i)
                    if(!(i >= (dimX-gpuDimX)/2 && i < (dimX-gpuDimX)/2+gpuDimX
                         && j >= (dimY-gpuDimY)/2 && j < (dimY-gpuDimY)/2+gpuDimY
                         && z >= (dimZ-gpuDimZ)/2 && z < (dimZ-gpuDimZ)/2+gpuDimZ))
                        datCPU[(z+cpuHaloWidth[4])*(dimX+cpuHaloWidth[0]+cpuHaloWidth[1])*(dimY+cpuHaloWidth[2]+cpuHaloWidth[3]) + (j+cpuHaloWidth[3])*(dimX+cpuHaloWidth[0]+cpuHaloWidth[1]) + i+cpuHaloWidth[0]] = z*dimX*dimY+j*dimX+i+1;
    }


    if(!cpuonly) {

        // how many points only on the device and an OpenCL buffer for them
        datGPU = new real_t[(gpuDimX+gpuHaloWidth[0]+gpuHaloWidth[1])*(gpuDimY+gpuHaloWidth[2]+gpuHaloWidth[3])*(gpuDimZ+gpuHaloWidth[4]+gpuHaloWidth[5])]{};

        for(int z = 0; z < gpuDimZ; ++z)
            for(int j = 0; j < gpuDimY; ++j)
                for(int i = 0; i < gpuDimX; ++i)
                    datGPU[(z+gpuHaloWidth[4])*(gpuDimX+gpuHaloWidth[0]+gpuHaloWidth[1])*(gpuDimY+gpuHaloWidth[2]+gpuHaloWidth[3]) + (j+gpuHaloWidth[3])*(gpuDimX+gpuHaloWidth[0]+gpuHaloWidth[1]) + i+gpuHaloWidth[0]] = z*gpuDimX*gpuDimY+j*gpuDimX+i+1;

        try {

            cl_datGpu = cl::Buffer(tausch->getContext(), &datGPU[0], (&datGPU[(gpuDimX+gpuHaloWidth[0]+gpuHaloWidth[1])*(gpuDimY+gpuHaloWidth[2]+gpuHaloWidth[3])*(gpuDimZ+gpuHaloWidth[4]+gpuHaloWidth[5])-1])+1, false);

        } catch(cl::Error error) {
            std::cout << "[sample] OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
            exit(1);
        }

    } else
        datGPU = new real_t[1]{};

    // pass pointers to the two data containers
    tausch->setCpuData(datCPU);
    if(!cpuonly)
        tausch->setGpuData(cl_datGpu, gpuDim);

}

Sample::~Sample() {

    delete tausch;
    delete[] datCPU;
    delete[] datGPU;

}

void Sample::launchCPU() {

    for(int run = 0; run < loops; ++run) {

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
        cl::copy(tausch->getQueue(), cl_datGpu, &datGPU[0], (&datGPU[(gpuDimX+gpuHaloWidth[0]+gpuHaloWidth[1])*(gpuDimY+gpuHaloWidth[2]+gpuHaloWidth[3])*(gpuDimZ+gpuHaloWidth[4]+gpuHaloWidth[5])-1])+1);
    } catch(cl::Error error) {
        std::cout << "[launchGPU] OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
        exit(1);
    }

}

void Sample::printCPU() {

    if(cpuonly) {

        for(int z = 0; z < dimZ+cpuHaloWidth[4]+cpuHaloWidth[5]; ++z) {

            std::cout << std::endl << "z = " << z << std::endl;

            for(int j = dimY+cpuHaloWidth[2]+cpuHaloWidth[3]-1; j >= 0; --j) {
                for(int i = 0; i < dimX+cpuHaloWidth[0]+cpuHaloWidth[1]; ++i)
                    std::cout << std::setw(3) << datCPU[z*(dimX+cpuHaloWidth[0]+cpuHaloWidth[1])*(dimY+cpuHaloWidth[2]+cpuHaloWidth[3]) + j*(dimX+cpuHaloWidth[0]+cpuHaloWidth[1]) + i] << " ";
                std::cout << std::endl;
            }

        }

    } else {

        for(int z = 0; z < dimZ+cpuHaloWidth[4]+cpuHaloWidth[5]; ++z) {

            std::cout << std::endl << "z = " << z << std::endl;

            for(int j = dimY+cpuHaloWidth[2]+cpuHaloWidth[3]-1; j >= 0; --j) {
                for(int i = 0; i < dimX+cpuHaloWidth[0]+cpuHaloWidth[1]; ++i) {
                    if(i >= (dimX-gpuDimX)/2+cpuHaloWidth[Tausch3D::LEFT]+gpuHaloWidth[Tausch3D::LEFT] && i < (dimX-gpuDimX)/2+cpuHaloWidth[Tausch3D::LEFT]+gpuDimX-gpuHaloWidth[Tausch3D::RIGHT]
                       && j >= (dimY-gpuDimY)/2+cpuHaloWidth[Tausch3D::BOTTOM]+gpuHaloWidth[Tausch3D::BOTTOM] && j < (dimY-gpuDimY)/2+cpuHaloWidth[Tausch3D::BOTTOM]+gpuDimY-gpuHaloWidth[Tausch3D::TOP]
                       && z >= (dimZ-gpuDimZ)/2+cpuHaloWidth[Tausch3D::FRONT]+gpuHaloWidth[Tausch3D::FRONT] && z < (dimZ-gpuDimZ)/2+cpuHaloWidth[Tausch3D::FRONT]+gpuDimZ-gpuHaloWidth[Tausch3D::BACK])
                        std::cout << std::setw(3) << "    ";
                    else
                        std::cout << std::setw(3) << datCPU[z*(dimX+cpuHaloWidth[0]+cpuHaloWidth[1])*(dimY+cpuHaloWidth[2]+cpuHaloWidth[3]) + j*(dimX+cpuHaloWidth[0]+cpuHaloWidth[1]) + i] << " ";
                }
                std::cout << std::endl;
            }

        }

    }

}

void Sample::printGPU() {

    for(int z = 0; z < gpuDimZ+gpuHaloWidth[Tausch3D::FRONT]+gpuHaloWidth[Tausch3D::BACK]; ++z) {

        std::cout << std::endl << "z = " << z << std::endl;

        for(int i = gpuDimY+gpuHaloWidth[Tausch3D::TOP]+gpuHaloWidth[Tausch3D::BOTTOM] -1; i >= 0; --i) {
            for(int j = 0; j < gpuDimX+gpuHaloWidth[Tausch3D::LEFT]+gpuHaloWidth[Tausch3D::RIGHT]; ++j)
                std::cout << std::setw(3) << datGPU[z*(gpuDimX+gpuHaloWidth[Tausch3D::LEFT]+gpuHaloWidth[Tausch3D::RIGHT])*(gpuDimY+gpuHaloWidth[Tausch3D::TOP]+gpuHaloWidth[Tausch3D::BOTTOM]) + i*(gpuDimX+gpuHaloWidth[Tausch3D::LEFT]+gpuHaloWidth[Tausch3D::RIGHT]) + j] << " ";
            std::cout << std::endl;
        }

    }

}

