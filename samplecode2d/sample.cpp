#include "sample.h"

Sample::Sample(size_t *localDim, size_t *gpuDim, size_t loops, size_t *cpuHaloWidth, size_t *gpuHaloWidth, size_t *cpuForGpuHaloWidth, size_t *mpiNum, bool buildlog, bool hybrid) {

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

    valuesPerPointPerBuffer = new size_t[numBuffers];
    for(int b = 0; b < numBuffers; ++b)
        valuesPerPointPerBuffer[b] = 1;

    dat = new double*[numBuffers];
    for(int b = 0; b < numBuffers; ++b)
        dat[b] = new double[valuesPerPointPerBuffer[b]*(localDim[0] + cpuHaloWidth[0] + cpuHaloWidth[1])*(localDim[1] + cpuHaloWidth[2] + cpuHaloWidth[3])]{};
    if(!hybrid) {
        for(int j = 0; j < localDim[1]; ++j)
            for(int i = 0; i < localDim[0]; ++i) {
                for(int b = 0; b < numBuffers; ++b)
                    for(int val = 0; val < valuesPerPointPerBuffer[b]; ++val)
                        dat[b][valuesPerPointPerBuffer[b]*((j+cpuHaloWidth[3])*(localDim[0]+cpuHaloWidth[0]+cpuHaloWidth[1]) + i+cpuHaloWidth[0])+val] =
                                (b*5 + j*localDim[0]+i+1)*10+val;
            }
    } else {
        for(int j = 0; j < localDim[1]; ++j)
            for(int i = 0; i < localDim[0]; ++i) {
                if(i >= (localDim[0]-gpuDim[0])/2 && i < (localDim[0]-gpuDim[0])/2+gpuDim[0] &&
                   j >= (localDim[1]-gpuDim[1])/2 && j < (localDim[1]-gpuDim[1])/2+gpuDim[1])
                    continue;
                for(int b = 0; b < numBuffers; ++b)
                    for(int val = 0; val < valuesPerPointPerBuffer[b]; ++val)
                        dat[b][valuesPerPointPerBuffer[b]*((j+cpuHaloWidth[3])*(localDim[0]+cpuHaloWidth[0]+cpuHaloWidth[1]) + i+cpuHaloWidth[0])+val] =
                                (b*5 + j*localDim[0]+i+1)*10+val;
            }
    }


    size_t tauschLocalDim[2] = {localDim[0]+cpuHaloWidth[0]+cpuHaloWidth[1], localDim[1]+cpuHaloWidth[2]+cpuHaloWidth[3]};
    tausch = new Tausch2D<double>(MPI_DOUBLE, numBuffers, valuesPerPointPerBuffer);

    // These are the (up to) 4 remote halos that are needed by this rank
    remoteHaloSpecsCpu = new TauschHaloSpec[4];
    // These are the (up to) 4 local halos that are needed tobe sent by this rank
    localHaloSpecsCpu = new TauschHaloSpec[4];

    localHaloSpecsCpu[0].bufferWidth = tauschLocalDim[0]; localHaloSpecsCpu[0].bufferHeight = tauschLocalDim[1];
    localHaloSpecsCpu[0].haloX = cpuHaloWidth[0]; localHaloSpecsCpu[0].haloY = 0;
    localHaloSpecsCpu[0].haloWidth = cpuHaloWidth[1]; localHaloSpecsCpu[0].haloHeight = cpuHaloWidth[3]+localDim[1]+cpuHaloWidth[2];
    localHaloSpecsCpu[0].remoteMpiRank = left;
    remoteHaloSpecsCpu[0].bufferWidth = tauschLocalDim[0]; remoteHaloSpecsCpu[0].bufferHeight = tauschLocalDim[1];
    remoteHaloSpecsCpu[0].haloX = 0; remoteHaloSpecsCpu[0].haloY = 0;
    remoteHaloSpecsCpu[0].haloWidth = cpuHaloWidth[0]; remoteHaloSpecsCpu[0].haloHeight = cpuHaloWidth[3]+localDim[1]+cpuHaloWidth[2];
    remoteHaloSpecsCpu[0].remoteMpiRank = left;

    localHaloSpecsCpu[1].bufferWidth = tauschLocalDim[0]; localHaloSpecsCpu[1].bufferHeight = tauschLocalDim[1];
    localHaloSpecsCpu[1].haloX = localDim[0]; localHaloSpecsCpu[1].haloY = 0;
    localHaloSpecsCpu[1].haloWidth = cpuHaloWidth[0]; localHaloSpecsCpu[1].haloHeight = cpuHaloWidth[3]+localDim[1]+cpuHaloWidth[2];
    localHaloSpecsCpu[1].remoteMpiRank = right;
    remoteHaloSpecsCpu[1].bufferWidth = tauschLocalDim[0]; remoteHaloSpecsCpu[1].bufferHeight = tauschLocalDim[1];
    remoteHaloSpecsCpu[1].haloX = cpuHaloWidth[0]+localDim[0]; remoteHaloSpecsCpu[1].haloY = 0;
    remoteHaloSpecsCpu[1].haloWidth = cpuHaloWidth[1]; remoteHaloSpecsCpu[1].haloHeight = cpuHaloWidth[3]+localDim[1]+cpuHaloWidth[2];
    remoteHaloSpecsCpu[1].remoteMpiRank = right;

    localHaloSpecsCpu[2].bufferWidth = tauschLocalDim[0]; localHaloSpecsCpu[2].bufferHeight = tauschLocalDim[1];
    localHaloSpecsCpu[2].haloX = 0; localHaloSpecsCpu[2].haloY = localDim[1];
    localHaloSpecsCpu[2].haloWidth = cpuHaloWidth[0]+localDim[0]+cpuHaloWidth[1]; localHaloSpecsCpu[2].haloHeight = cpuHaloWidth[3];
    localHaloSpecsCpu[2].remoteMpiRank = top;
    remoteHaloSpecsCpu[2].bufferWidth = tauschLocalDim[0]; remoteHaloSpecsCpu[2].bufferHeight = tauschLocalDim[1];
    remoteHaloSpecsCpu[2].haloX = 0; remoteHaloSpecsCpu[2].haloY = cpuHaloWidth[3]+localDim[1];
    remoteHaloSpecsCpu[2].haloWidth = cpuHaloWidth[0]+localDim[0]+cpuHaloWidth[1]; remoteHaloSpecsCpu[2].haloHeight = cpuHaloWidth[2];
    remoteHaloSpecsCpu[2].remoteMpiRank = top;

    localHaloSpecsCpu[3].bufferWidth = tauschLocalDim[0]; localHaloSpecsCpu[3].bufferHeight = tauschLocalDim[1];
    localHaloSpecsCpu[3].haloX = 0; localHaloSpecsCpu[3].haloY = cpuHaloWidth[3];
    localHaloSpecsCpu[3].haloWidth = cpuHaloWidth[0]+localDim[0]+cpuHaloWidth[1]; localHaloSpecsCpu[3].haloHeight = cpuHaloWidth[2];
    localHaloSpecsCpu[3].remoteMpiRank = bottom;
    remoteHaloSpecsCpu[3].bufferWidth = tauschLocalDim[0]; remoteHaloSpecsCpu[3].bufferHeight = tauschLocalDim[1];
    remoteHaloSpecsCpu[3].haloX = 0; remoteHaloSpecsCpu[3].haloY = 0;
    remoteHaloSpecsCpu[3].haloWidth = cpuHaloWidth[0]+localDim[0]+cpuHaloWidth[1]; remoteHaloSpecsCpu[3].haloHeight = cpuHaloWidth[3];
    remoteHaloSpecsCpu[3].remoteMpiRank = bottom;

    tausch->setLocalHaloInfoCpu(4, localHaloSpecsCpu);
    tausch->setRemoteHaloInfoCpu(4, remoteHaloSpecsCpu);

    if(hybrid) {

        size_t tauschGpuDim[2] = {gpuDim[0]+gpuHaloWidth[0]+gpuHaloWidth[1], gpuDim[1]+gpuHaloWidth[2]+gpuHaloWidth[3]};
        tausch->enableOpenCL(true, 64, true, buildlog);

        gpudat = new double*[numBuffers];
        for(int b = 0; b < numBuffers; ++b)
            gpudat[b] = new double[valuesPerPointPerBuffer[b]*(gpuDim[0] + gpuHaloWidth[0] + gpuHaloWidth[1])*(gpuDim[1] + gpuHaloWidth[2] + gpuHaloWidth[3])]{};

        for(int j = 0; j < gpuDim[1]; ++j)
            for(int i = 0; i < gpuDim[0]; ++i) {
                for(int b = 0; b < numBuffers; ++b)
                    for(int val = 0; val < valuesPerPointPerBuffer[b]; ++val)
                        gpudat[b][valuesPerPointPerBuffer[b]*((j+gpuHaloWidth[3])*(gpuDim[0]+gpuHaloWidth[0]+gpuHaloWidth[1]) + i+gpuHaloWidth[0])+val] =
                                (b*5 + j*gpuDim[0]+i+1)*10+val;
            }

        try {
            cl_gpudat = new cl::Buffer[numBuffers];
            for(int b = 0; b < numBuffers; ++b) {
                int s = valuesPerPointPerBuffer[b]*(gpuDim[0] + gpuHaloWidth[0] + gpuHaloWidth[1])*(gpuDim[1] + gpuHaloWidth[2] + gpuHaloWidth[3]);
                cl_gpudat[b] = cl::Buffer(tausch->getOpenCLContext(), &gpudat[b][0], (&gpudat[b][s-1])+1, false);
            }
        } catch(cl::Error error) {
            std::cerr << "Samplecode2D :: constructor :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
            exit(1);
        }

        remoteHaloSpecsCpuForGpu = new TauschHaloSpec[4];
        localHaloSpecsCpuForGpu = new TauschHaloSpec[4];

        remoteHaloSpecsCpuForGpu[0].bufferWidth = tauschLocalDim[0]; remoteHaloSpecsCpuForGpu[3].bufferHeight = tauschLocalDim[1];
        remoteHaloSpecsCpuForGpu[0].haloX = (localDim[0]-gpuDim[0])/2+cpuHaloWidth[0];
        remoteHaloSpecsCpuForGpu[0].haloY = (localDim[1]-gpuDim[1])/2+cpuHaloWidth[3];
        remoteHaloSpecsCpuForGpu[0].haloWidth = cpuForGpuHaloWidth[0];
        remoteHaloSpecsCpuForGpu[0].haloHeight = gpuDim[1];

        remoteHaloSpecsCpuForGpu[1].bufferWidth = tauschLocalDim[0]; remoteHaloSpecsCpuForGpu[3].bufferHeight = tauschLocalDim[1];
        remoteHaloSpecsCpuForGpu[1].haloX = (localDim[0]-gpuDim[0])/2+cpuHaloWidth[0]+gpuDim[0]-cpuForGpuHaloWidth[1];
        remoteHaloSpecsCpuForGpu[1].haloY = (localDim[1]-gpuDim[1])/2+cpuHaloWidth[3];
        remoteHaloSpecsCpuForGpu[1].haloWidth = cpuForGpuHaloWidth[1];
        remoteHaloSpecsCpuForGpu[1].haloHeight = gpuDim[1];

        remoteHaloSpecsCpuForGpu[2].bufferWidth = tauschLocalDim[0]; remoteHaloSpecsCpuForGpu[3].bufferHeight = tauschLocalDim[1];
        remoteHaloSpecsCpuForGpu[2].haloX = (localDim[0]-gpuDim[0])/2+cpuHaloWidth[0]+cpuForGpuHaloWidth[0];
        remoteHaloSpecsCpuForGpu[2].haloY = (localDim[1]-gpuDim[1])/2+cpuHaloWidth[3]+gpuDim[1]-cpuForGpuHaloWidth[2];
        remoteHaloSpecsCpuForGpu[2].haloWidth = gpuDim[0]-cpuForGpuHaloWidth[0]-cpuForGpuHaloWidth[1];
        remoteHaloSpecsCpuForGpu[2].haloHeight = cpuForGpuHaloWidth[2];

        remoteHaloSpecsCpuForGpu[3].bufferWidth = tauschLocalDim[0]; remoteHaloSpecsCpuForGpu[3].bufferHeight = tauschLocalDim[1];
        remoteHaloSpecsCpuForGpu[3].haloX = (localDim[0]-gpuDim[0])/2+cpuHaloWidth[0]+cpuForGpuHaloWidth[0];
        remoteHaloSpecsCpuForGpu[3].haloY = (localDim[1]-gpuDim[1])/2+cpuHaloWidth[3];
        remoteHaloSpecsCpuForGpu[3].haloWidth = gpuDim[0]-cpuForGpuHaloWidth[0]-cpuForGpuHaloWidth[1];
        remoteHaloSpecsCpuForGpu[3].haloHeight = cpuForGpuHaloWidth[3];

        localHaloSpecsCpuForGpu[0].bufferWidth = tauschLocalDim[0]; localHaloSpecsCpuForGpu[3].bufferHeight = tauschLocalDim[1];
        localHaloSpecsCpuForGpu[0].haloX = (localDim[0]-gpuDim[0])/2+cpuHaloWidth[0]-gpuHaloWidth[0];
        localHaloSpecsCpuForGpu[0].haloY = (localDim[1]-gpuDim[1])/2+cpuHaloWidth[3]-gpuHaloWidth[3];
        localHaloSpecsCpuForGpu[0].haloWidth = gpuHaloWidth[0];
        localHaloSpecsCpuForGpu[0].haloHeight = gpuDim[1]+gpuHaloWidth[2]+gpuHaloWidth[3];

        localHaloSpecsCpuForGpu[1].bufferWidth = tauschLocalDim[0]; localHaloSpecsCpuForGpu[3].bufferHeight = tauschLocalDim[1];
        localHaloSpecsCpuForGpu[1].haloX = (localDim[0]-gpuDim[0])/2+cpuHaloWidth[0]+gpuDim[0];
        localHaloSpecsCpuForGpu[1].haloY = (localDim[1]-gpuDim[1])/2+cpuHaloWidth[3]-gpuHaloWidth[3];
        localHaloSpecsCpuForGpu[1].haloWidth = gpuHaloWidth[1];
        localHaloSpecsCpuForGpu[1].haloHeight = gpuDim[1]+gpuHaloWidth[2]+gpuHaloWidth[3];

        localHaloSpecsCpuForGpu[2].bufferWidth = tauschLocalDim[0]; localHaloSpecsCpuForGpu[3].bufferHeight = tauschLocalDim[1];
        localHaloSpecsCpuForGpu[2].haloX = (localDim[0]-gpuDim[0])/2+cpuHaloWidth[0]-gpuHaloWidth[0];
        localHaloSpecsCpuForGpu[2].haloY = (localDim[1]-gpuDim[1])/2+cpuHaloWidth[3]+gpuDim[1];
        localHaloSpecsCpuForGpu[2].haloWidth = gpuDim[0]+gpuHaloWidth[0]+gpuHaloWidth[1];
        localHaloSpecsCpuForGpu[2].haloHeight = gpuHaloWidth[2];

        localHaloSpecsCpuForGpu[3].bufferWidth = tauschLocalDim[0]; localHaloSpecsCpuForGpu[3].bufferHeight = tauschLocalDim[1];
        localHaloSpecsCpuForGpu[3].haloX = (localDim[0]-gpuDim[0])/2+cpuHaloWidth[0]-gpuHaloWidth[0];
        localHaloSpecsCpuForGpu[3].haloY = (localDim[1]-gpuDim[1])/2+cpuHaloWidth[3]-gpuHaloWidth[3];
        localHaloSpecsCpuForGpu[3].haloWidth = gpuDim[0]+gpuHaloWidth[0]+gpuHaloWidth[1];
        localHaloSpecsCpuForGpu[3].haloHeight = gpuHaloWidth[3];

        tausch->setLocalHaloInfoCpuForGpu(4, localHaloSpecsCpuForGpu);
        tausch->setRemoteHaloInfoCpuForGpu(4, remoteHaloSpecsCpuForGpu);

        remoteHaloSpecsGpu = new TauschHaloSpec[4];
        localHaloSpecsGpu = new TauschHaloSpec[4];

        localHaloSpecsGpu[0].bufferWidth = tauschGpuDim[0]; localHaloSpecsGpu[0].bufferHeight = tauschGpuDim[1];
        localHaloSpecsGpu[0].haloX = gpuHaloWidth[0];
        localHaloSpecsGpu[0].haloY = gpuHaloWidth[3];
        localHaloSpecsGpu[0].haloWidth = cpuForGpuHaloWidth[0];
        localHaloSpecsGpu[0].haloHeight = gpuDim[1];

        localHaloSpecsGpu[1].bufferWidth = tauschGpuDim[0]; localHaloSpecsGpu[2].bufferHeight = tauschGpuDim[1];
        localHaloSpecsGpu[1].haloX = gpuDim[0]+gpuHaloWidth[0]-cpuForGpuHaloWidth[1];
        localHaloSpecsGpu[1].haloY = gpuHaloWidth[3];
        localHaloSpecsGpu[1].haloWidth = cpuForGpuHaloWidth[1];
        localHaloSpecsGpu[1].haloHeight = gpuDim[1];

        localHaloSpecsGpu[2].bufferWidth = tauschGpuDim[0]; localHaloSpecsGpu[1].bufferHeight = tauschGpuDim[1];
        localHaloSpecsGpu[2].haloX = gpuHaloWidth[0]+cpuForGpuHaloWidth[0];
        localHaloSpecsGpu[2].haloY = gpuDim[1]+gpuHaloWidth[3]-cpuForGpuHaloWidth[2];
        localHaloSpecsGpu[2].haloWidth = gpuDim[0] - cpuForGpuHaloWidth[0] - cpuForGpuHaloWidth[1];
        localHaloSpecsGpu[2].haloHeight = cpuForGpuHaloWidth[2];

        localHaloSpecsGpu[3].bufferWidth = tauschGpuDim[0]; localHaloSpecsGpu[3].bufferHeight = tauschGpuDim[1];
        localHaloSpecsGpu[3].haloX = gpuHaloWidth[0]+cpuForGpuHaloWidth[0];
        localHaloSpecsGpu[3].haloY = gpuHaloWidth[3];
        localHaloSpecsGpu[3].haloWidth = gpuDim[0] - cpuForGpuHaloWidth[0] - cpuForGpuHaloWidth[1];
        localHaloSpecsGpu[3].haloHeight = cpuForGpuHaloWidth[3];

        remoteHaloSpecsGpu[0].bufferWidth = tauschGpuDim[0]; remoteHaloSpecsGpu[0].bufferHeight = tauschGpuDim[1];
        remoteHaloSpecsGpu[0].haloX = 0;
        remoteHaloSpecsGpu[0].haloY = 0;
        remoteHaloSpecsGpu[0].haloWidth = gpuHaloWidth[0];
        remoteHaloSpecsGpu[0].haloHeight = gpuDim[1] + gpuHaloWidth[2]+gpuHaloWidth[3];

        remoteHaloSpecsGpu[1].bufferWidth = tauschGpuDim[0]; remoteHaloSpecsGpu[2].bufferHeight = tauschGpuDim[2];
        remoteHaloSpecsGpu[1].haloX = gpuDim[0]+gpuHaloWidth[0];
        remoteHaloSpecsGpu[1].haloY = 0;
        remoteHaloSpecsGpu[1].haloWidth = gpuHaloWidth[1];
        remoteHaloSpecsGpu[1].haloHeight = gpuDim[1] + gpuHaloWidth[2]+gpuHaloWidth[3];

        remoteHaloSpecsGpu[2].bufferWidth = tauschGpuDim[0]; remoteHaloSpecsGpu[1].bufferHeight = tauschGpuDim[1];
        remoteHaloSpecsGpu[2].haloX = 0;
        remoteHaloSpecsGpu[2].haloY = gpuDim[1]+gpuHaloWidth[3];
        remoteHaloSpecsGpu[2].haloWidth = gpuDim[0] + gpuHaloWidth[0]+gpuHaloWidth[1];
        remoteHaloSpecsGpu[2].haloHeight = gpuHaloWidth[2];

        remoteHaloSpecsGpu[3].bufferWidth = tauschGpuDim[0]; remoteHaloSpecsGpu[3].bufferHeight = tauschGpuDim[1];
        remoteHaloSpecsGpu[3].haloX = 0;
        remoteHaloSpecsGpu[3].haloY = 0;
        remoteHaloSpecsGpu[3].haloWidth = gpuDim[0] + gpuHaloWidth[0]+gpuHaloWidth[1];
        remoteHaloSpecsGpu[3].haloHeight = gpuHaloWidth[3];

        tausch->setLocalHaloInfoGpu(4, localHaloSpecsGpu);
        tausch->setRemoteHaloInfoGpu(4, remoteHaloSpecsGpu);

    }

}

Sample::~Sample() {

    delete[] localHaloSpecsCpu;
    delete[] remoteHaloSpecsCpu;
    delete tausch;
    for(int b = 0; b < numBuffers; ++b)
        delete[] dat[b];
    delete[] dat;

}

void Sample::launchCPU() {

    if(hybrid) {

        for(int iter = 0; iter < loops; ++iter) {

            int sendtagsCpu[4] = {0, 1, 2, 3};
            int recvtagsCpu[4] = {1, 0, 3, 2};
            int sendtagsGpu[4] = {0, 1, 2, 3};
            int recvtagsGpu[4] = {0, 1, 2, 3};

            tausch->postAllReceivesCpu(recvtagsCpu);

            for(int ver_hor = 0; ver_hor < 2; ++ver_hor) {

                for(int b = 0; b < numBuffers; ++b)
                    tausch->packSendBufferCpu(2*ver_hor, b, dat[b]);
                tausch->sendCpu(2*ver_hor, sendtagsCpu[2*ver_hor]);

                for(int b = 0; b < numBuffers; ++b)
                    tausch->packSendBufferCpu(2*ver_hor+1, b, dat[b]);
                tausch->sendCpu(2*ver_hor+1, sendtagsCpu[2*ver_hor +1]);

                for(int b = 0; b < numBuffers; ++b)
                    tausch->packSendBufferCpuToGpu(2*ver_hor, b, dat[b]);
                tausch->sendCpuToGpu(2*ver_hor, sendtagsGpu[2*ver_hor]);

                for(int b = 0; b < numBuffers; ++b)
                    tausch->packSendBufferCpuToGpu(2*ver_hor+1, b, dat[b]);
                tausch->sendCpuToGpu(2*ver_hor+1, sendtagsGpu[2*ver_hor +1]);

                tausch->recvCpu(2*ver_hor);
                for(int b = 0; b < numBuffers; ++b)
                    tausch->unpackRecvBufferCpu(2*ver_hor, b, dat[b]);

                tausch->recvCpu(2*ver_hor+1);
                for(int b = 0; b < numBuffers; ++b)
                    tausch->unpackRecvBufferCpu(2*ver_hor+1, b, dat[b]);

                tausch->recvGpuToCpu(2*ver_hor, recvtagsGpu[2*ver_hor]);
                for(int b = 0; b < numBuffers; ++b)
                    tausch->unpackRecvBufferGpuToCpu(2*ver_hor, b, dat[b]);

                tausch->recvGpuToCpu(2*ver_hor+1, recvtagsGpu[2*ver_hor +1]);
                for(int b = 0; b < numBuffers; ++b)
                    tausch->unpackRecvBufferGpuToCpu(2*ver_hor+1, b, dat[b]);

            }

        }

    } else {

        for(int iter = 0; iter < loops; ++iter) {

            int sendtags[4] = {0, 1, 2, 3};
            int recvtags[4] = {0, 1, 2, 3};

            tausch->postAllReceivesCpu(recvtags);

            for(int ver_hor = 0; ver_hor < 2; ++ver_hor) {

                for(int b = 0; b < numBuffers; ++b)
                    tausch->packSendBufferCpu(2*ver_hor, b, dat[b]);
                tausch->sendCpu(2*ver_hor, sendtags[2*ver_hor]);

                for(int b = 0; b < numBuffers; ++b)
                    tausch->packSendBufferCpu(2*ver_hor+1, b, dat[b]);
                tausch->sendCpu(2*ver_hor+1, sendtags[2*ver_hor +1]);

                tausch->recvCpu(2*ver_hor);
                for(int b = 0; b < numBuffers; ++b)
                    tausch->unpackRecvBufferCpu(2*ver_hor, b, dat[b]);

                tausch->recvCpu(2*ver_hor+1);
                for(int b = 0; b < numBuffers; ++b)
                    tausch->unpackRecvBufferCpu(2*ver_hor+1, b, dat[b]);

            }

        }

    }

}

void Sample::launchGPU() {

    for(int iter = 0; iter < loops; ++iter) {

        int sendtags[4] = {0, 1, 2, 3};
        int recvtags[4] = {0, 1, 2, 3};

        for(int ver_hor = 0; ver_hor < 2; ++ver_hor) {

            tausch->recvCpuToGpu(2*ver_hor, recvtags[2*ver_hor]);
            for(int b = 0; b < numBuffers; ++b)
                tausch->unpackRecvBufferCpuToGpu(2*ver_hor, b, cl_gpudat[b]);

            tausch->recvCpuToGpu(2*ver_hor+1, recvtags[2*ver_hor +1]);
            for(int b = 0; b < numBuffers; ++b)
                tausch->unpackRecvBufferCpuToGpu(2*ver_hor+1, b, cl_gpudat[b]);

            for(int b = 0; b < numBuffers; ++b)
                tausch->packSendBufferGpuToCpu(2*ver_hor, b, cl_gpudat[b]);
            tausch->sendGpuToCpu(2*ver_hor, sendtags[2*ver_hor]);

            for(int b = 0; b < numBuffers; ++b)
                tausch->packSendBufferGpuToCpu(2*ver_hor+1, b, cl_gpudat[b]);
            tausch->sendGpuToCpu(2*ver_hor+1, sendtags[2*ver_hor +1]);

        }

    }

    for(int b = 0; b < numBuffers; ++b) {
        int s = valuesPerPointPerBuffer[b]*(gpuDim[0] + gpuHaloWidth[0] + gpuHaloWidth[1])*(gpuDim[1] + gpuHaloWidth[2] + gpuHaloWidth[3]);
        cl::copy(tausch->getOpenCLQueue(), cl_gpudat[b], &gpudat[b][0], &gpudat[b][s]);
    }

}

void Sample::printCPU() {

    for(int j = localDim[1]+cpuHaloWidth[2]+cpuHaloWidth[3]-1; j >= 0; --j) {
        for(int b = 0; b < numBuffers; ++b) {
            for(int val = 0; val < valuesPerPointPerBuffer[b]; ++val) {
                for(int i = 0; i < localDim[0]+cpuHaloWidth[0]+cpuHaloWidth[1]; ++i)
                    std::cout << std::setw(4) << dat[b][valuesPerPointPerBuffer[b]*(j*(localDim[0]+cpuHaloWidth[0]+cpuHaloWidth[1]) + i) + val] << " ";
                if(val != valuesPerPointPerBuffer[b]-1)
                    std::cout << "   ";
            }
            if(b != numBuffers-1)
                std::cout << "          ";
        }
        std::cout << std::endl;
    }

}

void Sample::printGPU() {

    for(int j = gpuDim[1]+gpuHaloWidth[2]+gpuHaloWidth[3]-1; j >= 0; --j) {
        for(int b = 0; b < numBuffers; ++b) {
            for(int val = 0; val < valuesPerPointPerBuffer[b]; ++val) {
                for(int i = 0; i < gpuDim[0]+gpuHaloWidth[0]+gpuHaloWidth[1]; ++i)
                    std::cout << std::setw(4) << gpudat[b][valuesPerPointPerBuffer[b]*(j*(gpuDim[0]+gpuHaloWidth[0]+gpuHaloWidth[1]) + i) + val] << " ";
                if(val != valuesPerPointPerBuffer[b]-1)
                    std::cout << "   ";
            }
            if(b != numBuffers-1)
                std::cout << "          ";
        }
        std::cout << std::endl;
    }

}
