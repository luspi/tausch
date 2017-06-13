#include "sample.h"

Sample::Sample(size_t *localDim, size_t *gpuDim, size_t loops, size_t *cpuHaloWidth, size_t *gpuHaloWidth, size_t *cpuForGpuHaloWidth, size_t *mpiNum, bool hybrid) {

    this->hybrid = hybrid;
    this->localDim[0] = localDim[0];
    this->localDim[1] = localDim[1];
    this->gpuDim[0] = gpuDim[0];
    this->gpuDim[1] = gpuDim[1];
    this->loops = loops;
    for(int i = 0; i < 4; ++i)
        this->cpuHaloWidth[i] = cpuHaloWidth[i];
    for(int i = 0; i < 4; ++i)
        this->gpuHaloWidth[i] = gpuHaloWidth[i];
    for(int i = 0; i < 4; ++i)
        this->cpuForGpuHaloWidth[i] = cpuForGpuHaloWidth[i];
    this->mpiNum[0] = mpiNum[0];
    this->mpiNum[1] = mpiNum[1];

    int mpiRank = 0, mpiSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    left = mpiRank-1, right = mpiRank+1, top = mpiRank+mpiNum[0], bottom = mpiRank-mpiNum[0];
    if(mpiRank%mpiNum[0] == 0)
        left += mpiNum[0];
    if((mpiRank+1)%mpiNum[0] == 0)
        right -= mpiNum[0];
    if(mpiRank < mpiNum[0])
        bottom += mpiSize;
    if(mpiRank >= mpiSize-mpiNum[0])
        top -= mpiSize;

    numBuffers = 2;
    valuesPerPointPerBuffer[0] = 1;
    valuesPerPointPerBuffer[1] = 1;
    dat1 = new double[valuesPerPointPerBuffer[0]*(localDim[0] + cpuHaloWidth[0] + cpuHaloWidth[1])*(localDim[1] + cpuHaloWidth[2] + cpuHaloWidth[3])]{};
    dat2 = new double[valuesPerPointPerBuffer[1]*(localDim[0] + cpuHaloWidth[0] + cpuHaloWidth[1])*(localDim[1] + cpuHaloWidth[2] + cpuHaloWidth[3])]{};
    if(!hybrid) {
        for(int j = 0; j < localDim[1]; ++j)
            for(int i = 0; i < localDim[0]; ++i) {
                for(int val = 0; val < valuesPerPointPerBuffer[0]; ++val)
                    dat1[valuesPerPointPerBuffer[0]*((j+cpuHaloWidth[3])*(localDim[0]+cpuHaloWidth[0]+cpuHaloWidth[1]) + i+cpuHaloWidth[0])+val] =
                            (j*localDim[0]+i+1)*10+val;
                for(int val = 0; val < valuesPerPointPerBuffer[1]; ++val)
                    dat2[valuesPerPointPerBuffer[1]*((j+cpuHaloWidth[3])*(localDim[0]+cpuHaloWidth[0]+cpuHaloWidth[1]) + i+cpuHaloWidth[0])+val] =
                            (5+j*localDim[0]+i+1)*10+val;
            }
    } else {
        for(int j = 0; j < localDim[1]; ++j)
            for(int i = 0; i < localDim[0]; ++i) {
                if(i >= (localDim[0]-gpuDim[0])/2 && i < (localDim[0]-gpuDim[0])/2+gpuDim[0] &&
                   j >= (localDim[1]-gpuDim[1])/2 && j < (localDim[1]-gpuDim[1])/2+gpuDim[1])
                    continue;
                for(int val = 0; val < valuesPerPointPerBuffer[0]; ++val)
                    dat1[valuesPerPointPerBuffer[0]*((j+cpuHaloWidth[3])*(localDim[0]+cpuHaloWidth[0]+cpuHaloWidth[1]) + i+cpuHaloWidth[0])+val] =
                            (j*localDim[0]+i+1)*10+val;
                for(int val = 0; val < valuesPerPointPerBuffer[1]; ++val)
                    dat2[valuesPerPointPerBuffer[1]*((j+cpuHaloWidth[3])*(localDim[0]+cpuHaloWidth[0]+cpuHaloWidth[1]) + i+cpuHaloWidth[0])+val] =
                            (5+j*localDim[0]+i+1)*10+val;
            }
    }


    size_t tauschLocalDim[2] = {localDim[0]+cpuHaloWidth[0]+cpuHaloWidth[1], localDim[1]+cpuHaloWidth[2]+cpuHaloWidth[3]};
    tausch = new Tausch2D<double>(tauschLocalDim, MPI_DOUBLE, numBuffers, valuesPerPointPerBuffer);

    // These are the (up to) 4 remote halos that are needed by this rank
    remoteHaloSpecsCpu = new TauschHaloSpec[4];
    // These are the (up to) 4 local halos that are needed tobe sent by this rank
    localHaloSpecsCpu = new TauschHaloSpec[4];

    localHaloSpecsCpu[0].x = cpuHaloWidth[0]; localHaloSpecsCpu[0].y = 0;
    localHaloSpecsCpu[0].width = cpuHaloWidth[1]; localHaloSpecsCpu[0].height = cpuHaloWidth[3]+localDim[1]+cpuHaloWidth[2];
    localHaloSpecsCpu[0].remoteMpiRank = left;
    remoteHaloSpecsCpu[0].x = 0; remoteHaloSpecsCpu[0].y = 0;
    remoteHaloSpecsCpu[0].width = cpuHaloWidth[0]; remoteHaloSpecsCpu[0].height = cpuHaloWidth[3]+localDim[1]+cpuHaloWidth[2];
    remoteHaloSpecsCpu[0].remoteMpiRank = left;

    localHaloSpecsCpu[1].x = localDim[0]; localHaloSpecsCpu[1].y = 0;
    localHaloSpecsCpu[1].width = cpuHaloWidth[0]; localHaloSpecsCpu[1].height = cpuHaloWidth[3]+localDim[1]+cpuHaloWidth[2];
    localHaloSpecsCpu[1].remoteMpiRank = right;
    remoteHaloSpecsCpu[1].x = cpuHaloWidth[0]+localDim[0]; remoteHaloSpecsCpu[1].y = 0;
    remoteHaloSpecsCpu[1].width = cpuHaloWidth[1]; remoteHaloSpecsCpu[1].height = cpuHaloWidth[3]+localDim[1]+cpuHaloWidth[2];
    remoteHaloSpecsCpu[1].remoteMpiRank = right;

    localHaloSpecsCpu[2].x = 0; localHaloSpecsCpu[2].y = localDim[1];
    localHaloSpecsCpu[2].width = cpuHaloWidth[0]+localDim[0]+cpuHaloWidth[1]; localHaloSpecsCpu[2].height = cpuHaloWidth[3];
    localHaloSpecsCpu[2].remoteMpiRank = top;
    remoteHaloSpecsCpu[2].x = 0; remoteHaloSpecsCpu[2].y = cpuHaloWidth[3]+localDim[1];
    remoteHaloSpecsCpu[2].width = cpuHaloWidth[0]+localDim[0]+cpuHaloWidth[1]; remoteHaloSpecsCpu[2].height = cpuHaloWidth[2];
    remoteHaloSpecsCpu[2].remoteMpiRank = top;

    localHaloSpecsCpu[3].x = 0; localHaloSpecsCpu[3].y = cpuHaloWidth[3];
    localHaloSpecsCpu[3].width = cpuHaloWidth[0]+localDim[0]+cpuHaloWidth[1]; localHaloSpecsCpu[3].height = cpuHaloWidth[2];
    localHaloSpecsCpu[3].remoteMpiRank = bottom;
    remoteHaloSpecsCpu[3].x = 0; remoteHaloSpecsCpu[3].y = 0;
    remoteHaloSpecsCpu[3].width = cpuHaloWidth[0]+localDim[0]+cpuHaloWidth[1]; remoteHaloSpecsCpu[3].height = cpuHaloWidth[3];
    remoteHaloSpecsCpu[3].remoteMpiRank = bottom;

    tausch->setLocalHaloInfoCpu(4, localHaloSpecsCpu);
    tausch->setRemoteHaloInfoCpu(4, remoteHaloSpecsCpu);

    if(hybrid) {

        size_t tauschGpuDim[2] = {gpuDim[0]+gpuHaloWidth[0]+gpuHaloWidth[1], gpuDim[1]+gpuHaloWidth[2]+gpuHaloWidth[3]};
        tausch->enableOpenCL(tauschGpuDim, true, 64, true);

        size_t gpudat1size = valuesPerPointPerBuffer[0]*(gpuDim[0] + gpuHaloWidth[0] + gpuHaloWidth[1])*(gpuDim[1] + gpuHaloWidth[2] + gpuHaloWidth[3]);
        size_t gpudat2size = valuesPerPointPerBuffer[1]*(gpuDim[0] + gpuHaloWidth[0] + gpuHaloWidth[1])*(gpuDim[1] + gpuHaloWidth[2] + gpuHaloWidth[3]);
        gpudat1 = new double[gpudat1size]{};
        gpudat2 = new double[gpudat2size]{};
        for(int j = 0; j < gpuDim[1]; ++j)
            for(int i = 0; i < gpuDim[0]; ++i) {
                for(int val = 0; val < valuesPerPointPerBuffer[0]; ++val)
                    gpudat1[valuesPerPointPerBuffer[0]*((j+gpuHaloWidth[3])*(gpuDim[0]+gpuHaloWidth[0]+gpuHaloWidth[1]) + i+gpuHaloWidth[0])+val] =
                            (j*gpuDim[0]+i+1)*10+val;
                for(int val = 0; val < valuesPerPointPerBuffer[1]; ++val)
                    gpudat2[valuesPerPointPerBuffer[1]*((j+gpuHaloWidth[3])*(gpuDim[0]+gpuHaloWidth[0]+gpuHaloWidth[1]) + i+gpuHaloWidth[0])+val] =
                            (5+j*gpuDim[0]+i+1)*10+val;
            }

        try {
            cl_gpudat1 = cl::Buffer(tausch->getOpenCLContext(), &gpudat1[0], (&gpudat1[gpudat1size-1])+1, false);
            cl_gpudat2 = cl::Buffer(tausch->getOpenCLContext(), &gpudat2[0], (&gpudat2[gpudat2size-1])+1, false);
        } catch(cl::Error error) {
            std::cerr << "Samplecode2D :: constructor :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
            exit(1);
        }

        remoteHaloSpecsCpuForGpu = new TauschHaloSpec[4];
        localHaloSpecsCpuForGpu = new TauschHaloSpec[4];

        remoteHaloSpecsCpuForGpu[0].x = (localDim[0]-gpuDim[0])/2+cpuHaloWidth[0];
        remoteHaloSpecsCpuForGpu[0].y = (localDim[1]-gpuDim[1])/2+cpuHaloWidth[3];
        remoteHaloSpecsCpuForGpu[0].width = cpuForGpuHaloWidth[0];
        remoteHaloSpecsCpuForGpu[0].height = gpuDim[1];

        remoteHaloSpecsCpuForGpu[1].x = (localDim[0]-gpuDim[0])/2+cpuHaloWidth[0]+gpuDim[0]-cpuForGpuHaloWidth[1];
        remoteHaloSpecsCpuForGpu[1].y = (localDim[1]-gpuDim[1])/2+cpuHaloWidth[3];
        remoteHaloSpecsCpuForGpu[1].width = cpuForGpuHaloWidth[1];
        remoteHaloSpecsCpuForGpu[1].height = gpuDim[1];

        remoteHaloSpecsCpuForGpu[2].x = (localDim[0]-gpuDim[0])/2+cpuHaloWidth[0]+cpuForGpuHaloWidth[0];
        remoteHaloSpecsCpuForGpu[2].y = (localDim[1]-gpuDim[1])/2+cpuHaloWidth[3]+gpuDim[1]-cpuForGpuHaloWidth[2];
        remoteHaloSpecsCpuForGpu[2].width = gpuDim[0]-cpuForGpuHaloWidth[0]-cpuForGpuHaloWidth[1];
        remoteHaloSpecsCpuForGpu[2].height = cpuForGpuHaloWidth[2];

        remoteHaloSpecsCpuForGpu[3].x = (localDim[0]-gpuDim[0])/2+cpuHaloWidth[0]+cpuForGpuHaloWidth[0];
        remoteHaloSpecsCpuForGpu[3].y = (localDim[1]-gpuDim[1])/2+cpuHaloWidth[3];
        remoteHaloSpecsCpuForGpu[3].width = gpuDim[0]-cpuForGpuHaloWidth[0]-cpuForGpuHaloWidth[1];
        remoteHaloSpecsCpuForGpu[3].height = cpuForGpuHaloWidth[3];

        localHaloSpecsCpuForGpu[0].x = (localDim[0]-gpuDim[0])/2+cpuHaloWidth[0]-gpuHaloWidth[0];
        localHaloSpecsCpuForGpu[0].y = (localDim[1]-gpuDim[1])/2+cpuHaloWidth[3]-gpuHaloWidth[3];
        localHaloSpecsCpuForGpu[0].width = gpuHaloWidth[0];
        localHaloSpecsCpuForGpu[0].height = gpuDim[1]+gpuHaloWidth[2]+gpuHaloWidth[3];

        localHaloSpecsCpuForGpu[1].x = (localDim[0]-gpuDim[0])/2+cpuHaloWidth[0]+gpuDim[0];
        localHaloSpecsCpuForGpu[1].y = (localDim[1]-gpuDim[1])/2+cpuHaloWidth[3]-gpuHaloWidth[3];
        localHaloSpecsCpuForGpu[1].width = gpuHaloWidth[1];
        localHaloSpecsCpuForGpu[1].height = gpuDim[1]+gpuHaloWidth[2]+gpuHaloWidth[3];

        localHaloSpecsCpuForGpu[2].x = (localDim[0]-gpuDim[0])/2+cpuHaloWidth[0]-gpuHaloWidth[0];
        localHaloSpecsCpuForGpu[2].y = (localDim[1]-gpuDim[1])/2+cpuHaloWidth[3]+gpuDim[1];
        localHaloSpecsCpuForGpu[2].width = gpuDim[0]+gpuHaloWidth[0]+gpuHaloWidth[1];
        localHaloSpecsCpuForGpu[2].height = gpuHaloWidth[2];

        localHaloSpecsCpuForGpu[3].x = (localDim[0]-gpuDim[0])/2+cpuHaloWidth[0]-gpuHaloWidth[0];
        localHaloSpecsCpuForGpu[3].y = (localDim[1]-gpuDim[1])/2+cpuHaloWidth[3]-gpuHaloWidth[3];
        localHaloSpecsCpuForGpu[3].width = gpuDim[0]+gpuHaloWidth[0]+gpuHaloWidth[1];
        localHaloSpecsCpuForGpu[3].height = gpuHaloWidth[3];

        tausch->setLocalHaloInfoCpuForGpu(4, localHaloSpecsCpuForGpu);
        tausch->setRemoteHaloInfoCpuForGpu(4, remoteHaloSpecsCpuForGpu);

        remoteHaloSpecsGpu = new TauschHaloSpec[4];
        localHaloSpecsGpu = new TauschHaloSpec[4];

        localHaloSpecsGpu[0].x = gpuHaloWidth[0];
        localHaloSpecsGpu[0].y = gpuHaloWidth[3];
        localHaloSpecsGpu[0].width = cpuForGpuHaloWidth[0];
        localHaloSpecsGpu[0].height = gpuDim[1];

        localHaloSpecsGpu[1].x = gpuDim[0]+gpuHaloWidth[0]-cpuForGpuHaloWidth[1];
        localHaloSpecsGpu[1].y = gpuHaloWidth[3];
        localHaloSpecsGpu[1].width = cpuForGpuHaloWidth[1];
        localHaloSpecsGpu[1].height = gpuDim[1];

        localHaloSpecsGpu[2].x = gpuHaloWidth[0]+cpuForGpuHaloWidth[0];
        localHaloSpecsGpu[2].y = gpuDim[1]+gpuHaloWidth[3]-cpuForGpuHaloWidth[2];
        localHaloSpecsGpu[2].width = gpuDim[0] - cpuForGpuHaloWidth[0] - cpuForGpuHaloWidth[1];
        localHaloSpecsGpu[2].height = cpuForGpuHaloWidth[2];

        localHaloSpecsGpu[3].x = gpuHaloWidth[0]+cpuForGpuHaloWidth[0];
        localHaloSpecsGpu[3].y = gpuHaloWidth[3];
        localHaloSpecsGpu[3].width = gpuDim[0] - cpuForGpuHaloWidth[0] - cpuForGpuHaloWidth[1];
        localHaloSpecsGpu[3].height = cpuForGpuHaloWidth[3];

        remoteHaloSpecsGpu[0].x = 0;
        remoteHaloSpecsGpu[0].y = 0;
        remoteHaloSpecsGpu[0].width = gpuHaloWidth[0];
        remoteHaloSpecsGpu[0].height = gpuDim[1] + gpuHaloWidth[2]+gpuHaloWidth[3];

        remoteHaloSpecsGpu[1].x = gpuDim[0]+gpuHaloWidth[0];
        remoteHaloSpecsGpu[1].y = 0;
        remoteHaloSpecsGpu[1].width = gpuHaloWidth[1];
        remoteHaloSpecsGpu[1].height = gpuDim[1] + gpuHaloWidth[2]+gpuHaloWidth[3];

        remoteHaloSpecsGpu[2].x = 0;
        remoteHaloSpecsGpu[2].y = gpuDim[1]+gpuHaloWidth[3];
        remoteHaloSpecsGpu[2].width = gpuDim[0] + gpuHaloWidth[0]+gpuHaloWidth[1];
        remoteHaloSpecsGpu[2].height = gpuHaloWidth[2];

        remoteHaloSpecsGpu[3].x = 0;
        remoteHaloSpecsGpu[3].y = 0;
        remoteHaloSpecsGpu[3].width = gpuDim[0] + gpuHaloWidth[0]+gpuHaloWidth[1];
        remoteHaloSpecsGpu[3].height = gpuHaloWidth[3];

        tausch->setLocalHaloInfoGpu(4, localHaloSpecsGpu);
        tausch->setRemoteHaloInfoGpu(4, remoteHaloSpecsGpu);

    }

}

Sample::~Sample() {

    delete[] localHaloSpecsCpu;
    delete[] remoteHaloSpecsCpu;
    delete tausch;
    delete[] dat1;
    delete[] dat2;

}

void Sample::launchCPU() {

    if(hybrid) {

        for(int iter = 0; iter < loops; ++iter) {

            int sendtags[4] = {0, 1, 2, 3};
            int recvtags[4] = {1, 0, 3, 2};

            tausch->postAllReceivesCpu(recvtags);

            for(int ver_hor = 0; ver_hor < 2; ++ver_hor) {

                tausch->packNextSendBufferCpu(2*ver_hor, dat1);
                tausch->packNextSendBufferCpu(2*ver_hor, dat2);
                tausch->sendCpu(2*ver_hor, sendtags[2*ver_hor]);

                tausch->packNextSendBufferCpu(2*ver_hor+1, dat1);
                tausch->packNextSendBufferCpu(2*ver_hor+1, dat2);
                tausch->sendCpu(2*ver_hor+1, sendtags[2*ver_hor +1]);

                tausch->packNextSendBufferCpuToGpu(2*ver_hor, dat1);
                tausch->packNextSendBufferCpuToGpu(2*ver_hor, dat2);
                tausch->sendCpuToGpu(2*ver_hor);

                tausch->packNextSendBufferCpuToGpu(2*ver_hor+1, dat1);
                tausch->packNextSendBufferCpuToGpu(2*ver_hor+1, dat2);
                tausch->sendCpuToGpu(2*ver_hor+1);

                tausch->recvCpu(2*ver_hor);
                tausch->unpackNextRecvBufferCpu(2*ver_hor, dat1);
                tausch->unpackNextRecvBufferCpu(2*ver_hor, dat2);

                tausch->recvCpu(2*ver_hor+1);
                tausch->unpackNextRecvBufferCpu(2*ver_hor+1, dat1);
                tausch->unpackNextRecvBufferCpu(2*ver_hor+1, dat2);

                tausch->recvGpuToCpu(2*ver_hor);
                tausch->unpackNextRecvBufferGpuToCpu(2*ver_hor, dat1);
                tausch->unpackNextRecvBufferGpuToCpu(2*ver_hor, dat2);

                tausch->recvGpuToCpu(2*ver_hor+1);
                tausch->unpackNextRecvBufferGpuToCpu(2*ver_hor+1, dat1);
                tausch->unpackNextRecvBufferGpuToCpu(2*ver_hor+1, dat2);

            }

        }

    } else {

        for(int iter = 0; iter < loops; ++iter) {

            int sendtags[4] = {0, 1, 2, 3};
            int recvtags[4] = {1, 0, 3, 2};

            tausch->postAllReceivesCpu(recvtags);

            for(int ver_hor = 0; ver_hor < 2; ++ver_hor) {

                tausch->packNextSendBufferCpu(2*ver_hor, dat1);
                tausch->packNextSendBufferCpu(2*ver_hor, dat2);
                tausch->sendCpu(2*ver_hor, sendtags[2*ver_hor]);

                tausch->packNextSendBufferCpu(2*ver_hor+1, dat1);
                tausch->packNextSendBufferCpu(2*ver_hor+1, dat2);
                tausch->sendCpu(2*ver_hor+1, sendtags[2*ver_hor +1]);

                tausch->recvCpu(2*ver_hor);
                tausch->unpackNextRecvBufferCpu(2*ver_hor, dat1);
                tausch->unpackNextRecvBufferCpu(2*ver_hor, dat2);

                tausch->recvCpu(2*ver_hor+1);
                tausch->unpackNextRecvBufferCpu(2*ver_hor+1, dat1);
                tausch->unpackNextRecvBufferCpu(2*ver_hor+1, dat2);

            }

        }

    }

}

void Sample::launchGPU() {

    for(int iter = 0; iter < loops; ++iter) {

        for(int ver_hor = 0; ver_hor < 2; ++ver_hor) {

            tausch->recvCpuToGpu(2*ver_hor);
            tausch->unpackNextRecvBufferCpuToGpu(2*ver_hor, cl_gpudat1);
            tausch->unpackNextRecvBufferCpuToGpu(2*ver_hor, cl_gpudat2);

            tausch->recvCpuToGpu(2*ver_hor+1);
            tausch->unpackNextRecvBufferCpuToGpu(2*ver_hor+1, cl_gpudat1);
            tausch->unpackNextRecvBufferCpuToGpu(2*ver_hor+1, cl_gpudat2);

            tausch->packNextSendBufferGpuToCpu(2*ver_hor, cl_gpudat1);
            tausch->packNextSendBufferGpuToCpu(2*ver_hor, cl_gpudat2);
            tausch->sendGpuToCpu(2*ver_hor);

            tausch->packNextSendBufferGpuToCpu(2*ver_hor+1, cl_gpudat1);
            tausch->packNextSendBufferGpuToCpu(2*ver_hor+1, cl_gpudat2);
            tausch->sendGpuToCpu(2*ver_hor+1);

        }

    }

    int gpudat1size = valuesPerPointPerBuffer[0]*(gpuDim[0] + gpuHaloWidth[0] + gpuHaloWidth[1])*(gpuDim[1] + gpuHaloWidth[2] + gpuHaloWidth[3]);
    int gpudat2size = valuesPerPointPerBuffer[1]*(gpuDim[0] + gpuHaloWidth[0] + gpuHaloWidth[1])*(gpuDim[1] + gpuHaloWidth[2] + gpuHaloWidth[3]);

    cl::copy(tausch->getOpenCLQueue(), cl_gpudat1, &gpudat1[0], &gpudat1[gpudat1size]);
    cl::copy(tausch->getOpenCLQueue(), cl_gpudat2, &gpudat2[0], &gpudat2[gpudat2size]);

}

void Sample::printCPU() {

    for(int j = localDim[1]+cpuHaloWidth[2]+cpuHaloWidth[3]-1; j >= 0; --j) {
        for(int val = 0; val < valuesPerPointPerBuffer[0]; ++val) {
            for(int i = 0; i < localDim[0]+cpuHaloWidth[0]+cpuHaloWidth[1]; ++i)
                std::cout << std::setw(4) << dat1[valuesPerPointPerBuffer[0]*(j*(localDim[0]+cpuHaloWidth[0]+cpuHaloWidth[1]) + i) + val] << " ";
            if(val != valuesPerPointPerBuffer[0]-1)
                std::cout << "   ";
        }
        std::cout << "          ";
        for(int val = 0; val < valuesPerPointPerBuffer[1]; ++val) {
            for(int i = 0; i < localDim[0]+cpuHaloWidth[0]+cpuHaloWidth[1]; ++i)
                std::cout << std::setw(4) << dat2[valuesPerPointPerBuffer[1]*(j*(localDim[0]+cpuHaloWidth[0]+cpuHaloWidth[1]) + i) + val] << " ";
            if(val != valuesPerPointPerBuffer[1]-1)
                std::cout << "   ";
        }
        std::cout << std::endl;
    }

}

void Sample::printGPU() {

    for(int j = gpuDim[1]+gpuHaloWidth[2]+gpuHaloWidth[3]-1; j >= 0; --j) {
        for(int val = 0; val < valuesPerPointPerBuffer[0]; ++val) {
            for(int i = 0; i < gpuDim[0]+gpuHaloWidth[0]+gpuHaloWidth[1]; ++i)
                std::cout << std::setw(4) << gpudat1[valuesPerPointPerBuffer[0]*(j*(gpuDim[0]+gpuHaloWidth[0]+gpuHaloWidth[1]) + i) + val] << " ";
            if(val != valuesPerPointPerBuffer[0]-1)
                std::cout << "   ";
        }
        std::cout << "          ";
        for(int val = 0; val < valuesPerPointPerBuffer[1]; ++val) {
            for(int i = 0; i < gpuDim[0]+gpuHaloWidth[0]+gpuHaloWidth[1]; ++i)
                std::cout << std::setw(4) << gpudat2[valuesPerPointPerBuffer[1]*(j*(gpuDim[0]+gpuHaloWidth[0]+gpuHaloWidth[1]) + i) + val] << " ";
            if(val != valuesPerPointPerBuffer[1]-1)
                std::cout << "   ";
        }
        std::cout << std::endl;
    }

}
