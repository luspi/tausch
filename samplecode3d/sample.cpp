#include "sample.h"

Sample::Sample(size_t *localDim, size_t *gpuDim, size_t loops, size_t *cpuHaloWidth, size_t *gpuHaloWidth, size_t *cpuForGpuHaloWidth, size_t *mpiNum, bool buildlog, bool hybrid) {

    for(int i = 0; i < 3; ++i)
        this->localDim[i] = localDim[i];
    for(int i = 0; i < 3; ++i)
        this->gpuDim[i] = gpuDim[i];
    this->loops = loops;
    for(int i = 0; i < 6; ++i)
        this->cpuHaloWidth[i] = cpuHaloWidth[i];
    for(int i = 0; i < 6; ++i)
        this->gpuHaloWidth[i] = gpuHaloWidth[i];
    for(int i = 0; i < 6; ++i)
        this->cpuForGpuHaloWidth[i] = cpuForGpuHaloWidth[i];
    for(int i = 0; i < 3; ++i)
        this->mpiNum[i] = mpiNum[i];

    this->hybrid = hybrid;

    int mpiRank = 0, mpiSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    left = mpiRank-1, right = mpiRank+1;
    top = mpiRank+mpiNum[TAUSCH_X], bottom = mpiRank-mpiNum[TAUSCH_X];
    front = mpiRank-mpiNum[TAUSCH_X]*mpiNum[TAUSCH_Y]; back = mpiRank+mpiNum[TAUSCH_X]*mpiNum[TAUSCH_Y];

    if(mpiRank%mpiNum[TAUSCH_X] == 0)
        left += mpiNum[TAUSCH_X];
    if((mpiRank+1)%mpiNum[TAUSCH_X] == 0)
        right -= mpiNum[TAUSCH_X];
    if(mpiRank%(mpiNum[TAUSCH_X]*mpiNum[TAUSCH_Y]) >= (mpiNum[TAUSCH_X]*mpiNum[TAUSCH_Y]-mpiNum[TAUSCH_X]))
        top -= mpiNum[TAUSCH_X]*mpiNum[TAUSCH_Y];
    if(mpiRank%(mpiNum[TAUSCH_X]*mpiNum[TAUSCH_Y]) < mpiNum[TAUSCH_X])
        bottom += mpiNum[TAUSCH_X]*mpiNum[TAUSCH_Y];
    if(mpiRank < mpiNum[TAUSCH_X]*mpiNum[TAUSCH_Y])
        front += mpiSize;
    if(mpiRank >= mpiSize-mpiNum[TAUSCH_X]*mpiNum[TAUSCH_Y])
        back -= mpiSize;

    numBuffers = 2;
    valuesPerPointPerBuffer = new size_t[numBuffers];
    for(int b = 0; b < numBuffers; ++b)
        valuesPerPointPerBuffer[b] = 1;

    dat = new double*[numBuffers];
    for(int b = 0; b < numBuffers; ++b)
        dat[b] = new double[valuesPerPointPerBuffer[b]*(localDim[0] + cpuHaloWidth[0] + cpuHaloWidth[1])*
                                              (localDim[1] + cpuHaloWidth[2] + cpuHaloWidth[3])*
                                              (localDim[2] + cpuHaloWidth[4] + cpuHaloWidth[5])]{};

    if(!hybrid) {
        for(int k = 0; k < localDim[TAUSCH_Z]; ++k)
            for(int j = 0; j < localDim[TAUSCH_Y]; ++j)
                for(int i = 0; i < localDim[TAUSCH_X]; ++i) {
                    for(int b = 0; b < numBuffers; ++b)
                        for(int val = 0; val < valuesPerPointPerBuffer[b]; ++val)
                            dat[b][valuesPerPointPerBuffer[b]*((k+cpuHaloWidth[4])*(localDim[TAUSCH_X]+cpuHaloWidth[0]+cpuHaloWidth[1])*
                                                 (localDim[TAUSCH_Y]+cpuHaloWidth[2]+cpuHaloWidth[3]) +
                                                 (j+cpuHaloWidth[3])*(localDim[TAUSCH_X]+cpuHaloWidth[0]+cpuHaloWidth[1]) +
                                                  i+cpuHaloWidth[0]) + val] = (b*5 + k*localDim[TAUSCH_X]*localDim[TAUSCH_Y] + j*localDim[TAUSCH_X] + i + 1)*10+val;
            }
    } else {
        for(int k = 0; k < localDim[TAUSCH_Z]; ++k)
            for(int j = 0; j < localDim[TAUSCH_Y]; ++j)
                for(int i = 0; i < localDim[TAUSCH_X]; ++i) {
                    if(i >= (localDim[0]-gpuDim[0])/2 && i < (localDim[0]-gpuDim[0])/2+gpuDim[0] &&
                       j >= (localDim[1]-gpuDim[1])/2 && j < (localDim[1]-gpuDim[1])/2+gpuDim[1] &&
                       k >= (localDim[2]-gpuDim[2])/2 && k < (localDim[2]-gpuDim[2])/2+gpuDim[2])
                        continue;
                    for(int b = 0; b < numBuffers; ++b)
                        for(int val = 0; val < valuesPerPointPerBuffer[b]; ++val)
                            dat[b][valuesPerPointPerBuffer[b]*((k+cpuHaloWidth[4])*(localDim[TAUSCH_X]+cpuHaloWidth[0]+cpuHaloWidth[1])*
                                                 (localDim[TAUSCH_Y]+cpuHaloWidth[2]+cpuHaloWidth[3]) +
                                                 (j+cpuHaloWidth[3])*(localDim[TAUSCH_X]+cpuHaloWidth[0]+cpuHaloWidth[1]) +
                                                  i+cpuHaloWidth[0]) + val] = (b*5 + k*localDim[TAUSCH_X]*localDim[TAUSCH_Y] + j*localDim[TAUSCH_X] + i + 1)*10+val;
            }
    }

    size_t tauschLocalDim[3] = {localDim[0]+cpuHaloWidth[0]+cpuHaloWidth[1],
                                localDim[1]+cpuHaloWidth[2]+cpuHaloWidth[3],
                                localDim[2]+cpuHaloWidth[4]+cpuHaloWidth[5]};
    tausch = new Tausch3D<double>(MPI_DOUBLE, numBuffers, valuesPerPointPerBuffer, MPI_COMM_WORLD);

    // These are the (up to) 4 remote halos that are needed by this rank
    remoteHaloSpecs = new TauschHaloSpec[6];
    // These are the (up to) 4 local halos that are needed tobe sent by this rank
    localHaloSpecs = new TauschHaloSpec[6];

    localHaloSpecs[0].bufferWidth = tauschLocalDim[0]; localHaloSpecs[0].bufferHeight = tauschLocalDim[1]; localHaloSpecs[0].bufferDepth = tauschLocalDim[2];
    localHaloSpecs[0].haloX = cpuHaloWidth[0]; localHaloSpecs[0].haloY = 0; localHaloSpecs[0].haloZ = 0;
    localHaloSpecs[0].haloWidth = cpuHaloWidth[1]; localHaloSpecs[0].haloHeight = cpuHaloWidth[3]+localDim[1]+cpuHaloWidth[2];
    localHaloSpecs[0].haloDepth = cpuHaloWidth[4]+localDim[2]+cpuHaloWidth[5]; localHaloSpecs[0].remoteMpiRank = left;

    localHaloSpecs[1].bufferWidth = tauschLocalDim[0]; localHaloSpecs[1].bufferHeight = tauschLocalDim[1]; localHaloSpecs[1].bufferDepth = tauschLocalDim[2];
    localHaloSpecs[1].haloX = localDim[0]; localHaloSpecs[1].haloY = 0; localHaloSpecs[1].haloZ = 0;
    localHaloSpecs[1].haloWidth = cpuHaloWidth[0]; localHaloSpecs[1].haloHeight = cpuHaloWidth[3]+localDim[1]+cpuHaloWidth[2];
    localHaloSpecs[1].haloDepth = cpuHaloWidth[4]+localDim[2]+cpuHaloWidth[5]; localHaloSpecs[1].remoteMpiRank = right;

    localHaloSpecs[2].bufferWidth = tauschLocalDim[0]; localHaloSpecs[2].bufferHeight = tauschLocalDim[1]; localHaloSpecs[2].bufferDepth = tauschLocalDim[2];
    localHaloSpecs[2].haloX = 0; localHaloSpecs[2].haloY = localDim[1]; localHaloSpecs[2].haloZ = 0;
    localHaloSpecs[2].haloWidth = cpuHaloWidth[0]+localDim[0]+cpuHaloWidth[1]; localHaloSpecs[2].haloHeight = cpuHaloWidth[3];
    localHaloSpecs[2].haloDepth = cpuHaloWidth[4]+localDim[2]+cpuHaloWidth[5]; localHaloSpecs[2].remoteMpiRank = top;

    localHaloSpecs[3].bufferWidth = tauschLocalDim[0]; localHaloSpecs[3].bufferHeight = tauschLocalDim[1]; localHaloSpecs[3].bufferDepth = tauschLocalDim[2];
    localHaloSpecs[3].haloX = 0; localHaloSpecs[3].haloY = cpuHaloWidth[3]; localHaloSpecs[3].haloZ = 0;
    localHaloSpecs[3].haloWidth = cpuHaloWidth[0]+localDim[0]+cpuHaloWidth[1]; localHaloSpecs[3].haloHeight = cpuHaloWidth[2];
    localHaloSpecs[3].haloDepth = cpuHaloWidth[4]+localDim[2]+cpuHaloWidth[5]; localHaloSpecs[3].remoteMpiRank = bottom;

    localHaloSpecs[4].bufferWidth = tauschLocalDim[0]; localHaloSpecs[4].bufferHeight = tauschLocalDim[1]; localHaloSpecs[4].bufferDepth = tauschLocalDim[2];
    localHaloSpecs[4].haloX = 0; localHaloSpecs[4].haloY = 0; localHaloSpecs[4].haloZ = cpuHaloWidth[4];
    localHaloSpecs[4].haloWidth = cpuHaloWidth[0]+localDim[0]+cpuHaloWidth[1]; localHaloSpecs[4].haloHeight = cpuHaloWidth[2]+localDim[1]+cpuHaloWidth[3];
    localHaloSpecs[4].haloDepth = cpuHaloWidth[5]; localHaloSpecs[4].remoteMpiRank = front;

    localHaloSpecs[5].bufferWidth = tauschLocalDim[0]; localHaloSpecs[5].bufferHeight = tauschLocalDim[1]; localHaloSpecs[5].bufferDepth = tauschLocalDim[2];
    localHaloSpecs[5].haloX = 0; localHaloSpecs[5].haloY = 0; localHaloSpecs[5].haloZ = localDim[2];
    localHaloSpecs[5].haloWidth = cpuHaloWidth[0]+localDim[0]+cpuHaloWidth[1]; localHaloSpecs[5].haloHeight = cpuHaloWidth[2]+localDim[1]+cpuHaloWidth[3];
    localHaloSpecs[5].haloDepth = cpuHaloWidth[4]; localHaloSpecs[5].remoteMpiRank = back;


    remoteHaloSpecs[0].bufferWidth = tauschLocalDim[0]; remoteHaloSpecs[0].bufferHeight = tauschLocalDim[1]; remoteHaloSpecs[0].bufferDepth = tauschLocalDim[2];
    remoteHaloSpecs[0].haloX = 0; remoteHaloSpecs[0].haloY = 0; remoteHaloSpecs[0].haloZ = 0;
    remoteHaloSpecs[0].haloWidth = cpuHaloWidth[0]; remoteHaloSpecs[0].haloHeight = cpuHaloWidth[3]+localDim[1]+cpuHaloWidth[2];
    remoteHaloSpecs[0].haloDepth = cpuHaloWidth[4]+localDim[2]+cpuHaloWidth[5]; remoteHaloSpecs[0].remoteMpiRank = left;

    remoteHaloSpecs[1].bufferWidth = tauschLocalDim[0]; remoteHaloSpecs[1].bufferHeight = tauschLocalDim[1]; remoteHaloSpecs[1].bufferDepth = tauschLocalDim[2];
    remoteHaloSpecs[1].haloX = localDim[0]+cpuHaloWidth[0]; remoteHaloSpecs[1].haloY = 0; remoteHaloSpecs[1].haloZ = 0;
    remoteHaloSpecs[1].haloWidth = cpuHaloWidth[1]; remoteHaloSpecs[1].haloHeight = cpuHaloWidth[3]+localDim[1]+cpuHaloWidth[2];
    remoteHaloSpecs[1].haloDepth = cpuHaloWidth[4]+localDim[2]+cpuHaloWidth[5]; remoteHaloSpecs[1].remoteMpiRank = right;

    remoteHaloSpecs[2].bufferWidth = tauschLocalDim[0]; remoteHaloSpecs[2].bufferHeight = tauschLocalDim[1]; remoteHaloSpecs[2].bufferDepth = tauschLocalDim[2];
    remoteHaloSpecs[2].haloX = 0; remoteHaloSpecs[2].haloY = localDim[1]+cpuHaloWidth[3]; remoteHaloSpecs[2].haloZ = 0;
    remoteHaloSpecs[2].haloWidth = cpuHaloWidth[0]+localDim[0]+cpuHaloWidth[1]; remoteHaloSpecs[2].haloHeight = cpuHaloWidth[2];
    remoteHaloSpecs[2].haloDepth = cpuHaloWidth[4]+localDim[2]+cpuHaloWidth[5]; remoteHaloSpecs[2].remoteMpiRank = top;

    remoteHaloSpecs[3].bufferWidth = tauschLocalDim[0]; remoteHaloSpecs[3].bufferHeight = tauschLocalDim[1]; remoteHaloSpecs[3].bufferDepth = tauschLocalDim[2];
    remoteHaloSpecs[3].haloX = 0; remoteHaloSpecs[3].haloY = 0; remoteHaloSpecs[3].haloZ = 0;
    remoteHaloSpecs[3].haloWidth = cpuHaloWidth[0]+localDim[0]+cpuHaloWidth[1]; remoteHaloSpecs[3].haloHeight = cpuHaloWidth[3];
    remoteHaloSpecs[3].haloDepth = cpuHaloWidth[4]+localDim[2]+cpuHaloWidth[5]; remoteHaloSpecs[3].remoteMpiRank = bottom;

    remoteHaloSpecs[4].bufferWidth = tauschLocalDim[0]; remoteHaloSpecs[4].bufferHeight = tauschLocalDim[1]; remoteHaloSpecs[4].bufferDepth = tauschLocalDim[2];
    remoteHaloSpecs[4].haloX = 0; remoteHaloSpecs[4].haloY = 0; remoteHaloSpecs[4].haloZ = 0;
    remoteHaloSpecs[4].haloWidth = cpuHaloWidth[0]+localDim[0]+cpuHaloWidth[1]; remoteHaloSpecs[4].haloHeight = cpuHaloWidth[2]+localDim[1]+cpuHaloWidth[3];
    remoteHaloSpecs[4].haloDepth = cpuHaloWidth[4]; remoteHaloSpecs[4].remoteMpiRank = front;

    remoteHaloSpecs[5].bufferWidth = tauschLocalDim[0]; remoteHaloSpecs[5].bufferHeight = tauschLocalDim[1]; remoteHaloSpecs[5].bufferDepth = tauschLocalDim[2];
    remoteHaloSpecs[5].haloX = 0; remoteHaloSpecs[5].haloY = 0; remoteHaloSpecs[5].haloZ = localDim[2]+cpuHaloWidth[4];
    remoteHaloSpecs[5].haloWidth = cpuHaloWidth[0]+localDim[0]+cpuHaloWidth[1]; remoteHaloSpecs[5].haloHeight = cpuHaloWidth[2]+localDim[1]+cpuHaloWidth[3];
    remoteHaloSpecs[5].haloDepth = cpuHaloWidth[5]; remoteHaloSpecs[5].remoteMpiRank = back;


    tausch->setLocalHaloInfo(TAUSCH_CPU|TAUSCH_WITHCPU, 6, localHaloSpecs);
    tausch->setRemoteHaloInfo(TAUSCH_CPU|TAUSCH_WITHCPU, 6, remoteHaloSpecs);


    if(hybrid) {

        size_t tauschGpuDim[3] = {gpuDim[0]+gpuHaloWidth[0]+gpuHaloWidth[1], gpuDim[1]+gpuHaloWidth[2]+gpuHaloWidth[3], gpuDim[2]+gpuHaloWidth[4]+gpuHaloWidth[5]};
        tausch->enableOpenCL(true, 64, true, buildlog);

        gpudat = new double*[numBuffers];
        for(int b = 0; b < numBuffers; ++b)
            gpudat[b] = new double[valuesPerPointPerBuffer[b]*(gpuDim[0] + gpuHaloWidth[0] + gpuHaloWidth[1])*(gpuDim[1] + gpuHaloWidth[2] + gpuHaloWidth[3])*(gpuDim[2] + gpuHaloWidth[4] + gpuHaloWidth[5])]{};

        for(int k = 0; k < gpuDim[2]; ++k)
            for(int j = 0; j < gpuDim[1]; ++j)
                for(int i = 0; i < gpuDim[0]; ++i) {
                    for(int b = 0; b < numBuffers; ++b)
                        for(int val = 0; val < valuesPerPointPerBuffer[b]; ++val)
                            gpudat[b][valuesPerPointPerBuffer[b]*((k+gpuHaloWidth[4])*(gpuDim[TAUSCH_X]+gpuHaloWidth[0]+gpuHaloWidth[1])*
                                                 (gpuDim[TAUSCH_Y]+gpuHaloWidth[2]+gpuHaloWidth[3]) +
                                                 (j+gpuHaloWidth[3])*(gpuDim[TAUSCH_X]+gpuHaloWidth[0]+gpuHaloWidth[1]) +
                                                  i+gpuHaloWidth[0]) + val] = (b*5 + k*gpuDim[TAUSCH_X]*gpuDim[TAUSCH_Y] + j*gpuDim[TAUSCH_X] + i + 1)*10+val;
            }

        try {
            cl_gpudat = new cl::Buffer[numBuffers];
            for(int b = 0; b < numBuffers; ++b) {
                int s = valuesPerPointPerBuffer[b]*(gpuDim[0] + gpuHaloWidth[0] + gpuHaloWidth[1])*(gpuDim[1] + gpuHaloWidth[2] + gpuHaloWidth[3])*(gpuDim[2] + gpuHaloWidth[4] + gpuHaloWidth[5]);
                cl_gpudat[b] = cl::Buffer(tausch->getOpenCLContext(), &gpudat[b][0], (&gpudat[b][s-1])+1, false);
            }
        } catch(cl::Error error) {
            std::cerr << "Samplecode2D :: constructor :: OpenCL exception caught: " << error.what() << " (" << error.err() << ")" << std::endl;
            exit(1);
        }

        remoteHaloSpecsCpuForGpu = new TauschHaloSpec[6];
        localHaloSpecsCpuForGpu = new TauschHaloSpec[6];

        remoteHaloSpecsCpuForGpu[0].bufferWidth = tauschLocalDim[0];
        remoteHaloSpecsCpuForGpu[0].bufferHeight = tauschLocalDim[1];
        remoteHaloSpecsCpuForGpu[0].bufferDepth = tauschLocalDim[2];
        remoteHaloSpecsCpuForGpu[0].haloX = (localDim[0]-gpuDim[0])/2+cpuHaloWidth[0];
        remoteHaloSpecsCpuForGpu[0].haloY = (localDim[1]-gpuDim[1])/2+cpuHaloWidth[3];
        remoteHaloSpecsCpuForGpu[0].haloZ = (localDim[2]-gpuDim[2])/2+cpuHaloWidth[4];
        remoteHaloSpecsCpuForGpu[0].haloWidth = cpuForGpuHaloWidth[0];
        remoteHaloSpecsCpuForGpu[0].haloHeight = gpuDim[1];
        remoteHaloSpecsCpuForGpu[0].haloDepth = gpuDim[2];

        remoteHaloSpecsCpuForGpu[1].bufferWidth = tauschLocalDim[0];
        remoteHaloSpecsCpuForGpu[1].bufferHeight = tauschLocalDim[1];
        remoteHaloSpecsCpuForGpu[1].bufferDepth = tauschLocalDim[2];
        remoteHaloSpecsCpuForGpu[1].haloX = (localDim[0]-gpuDim[0])/2+cpuHaloWidth[0]+gpuDim[0]-cpuForGpuHaloWidth[1];
        remoteHaloSpecsCpuForGpu[1].haloY = (localDim[1]-gpuDim[1])/2+cpuHaloWidth[3];
        remoteHaloSpecsCpuForGpu[1].haloZ = (localDim[2]-gpuDim[2])/2+cpuHaloWidth[4];
        remoteHaloSpecsCpuForGpu[1].haloWidth = cpuForGpuHaloWidth[1];
        remoteHaloSpecsCpuForGpu[1].haloHeight = gpuDim[1];
        remoteHaloSpecsCpuForGpu[1].haloDepth = gpuDim[2];

        remoteHaloSpecsCpuForGpu[2].bufferWidth = tauschLocalDim[0];
        remoteHaloSpecsCpuForGpu[2].bufferHeight = tauschLocalDim[1];
        remoteHaloSpecsCpuForGpu[2].bufferDepth = tauschLocalDim[2];
        remoteHaloSpecsCpuForGpu[2].haloX = (localDim[0]-gpuDim[0])/2+cpuHaloWidth[0]+cpuForGpuHaloWidth[0];
        remoteHaloSpecsCpuForGpu[2].haloY = (localDim[1]-gpuDim[1])/2+cpuHaloWidth[3]+gpuDim[1]-cpuForGpuHaloWidth[2];
        remoteHaloSpecsCpuForGpu[2].haloZ = (localDim[2]-gpuDim[2])/2+cpuHaloWidth[4];
        remoteHaloSpecsCpuForGpu[2].haloWidth = gpuDim[0]-cpuForGpuHaloWidth[0]-cpuForGpuHaloWidth[1];
        remoteHaloSpecsCpuForGpu[2].haloHeight = cpuForGpuHaloWidth[2];
        remoteHaloSpecsCpuForGpu[2].haloDepth = gpuDim[2];

        remoteHaloSpecsCpuForGpu[3].bufferWidth = tauschLocalDim[0];
        remoteHaloSpecsCpuForGpu[3].bufferHeight = tauschLocalDim[1];
        remoteHaloSpecsCpuForGpu[3].bufferDepth = tauschLocalDim[2];
        remoteHaloSpecsCpuForGpu[3].haloX = (localDim[0]-gpuDim[0])/2+cpuHaloWidth[0]+cpuForGpuHaloWidth[0];
        remoteHaloSpecsCpuForGpu[3].haloY = (localDim[1]-gpuDim[1])/2+cpuHaloWidth[3];
        remoteHaloSpecsCpuForGpu[3].haloZ = (localDim[2]-gpuDim[2])/2+cpuHaloWidth[4];
        remoteHaloSpecsCpuForGpu[3].haloWidth = gpuDim[0]-cpuForGpuHaloWidth[0]-cpuForGpuHaloWidth[1];
        remoteHaloSpecsCpuForGpu[3].haloHeight = cpuForGpuHaloWidth[3];
        remoteHaloSpecsCpuForGpu[3].haloDepth = gpuDim[2];

        remoteHaloSpecsCpuForGpu[4].bufferWidth = tauschLocalDim[0];
        remoteHaloSpecsCpuForGpu[4].bufferHeight = tauschLocalDim[1];
        remoteHaloSpecsCpuForGpu[4].bufferDepth = tauschLocalDim[2];
        remoteHaloSpecsCpuForGpu[4].haloX = (localDim[0]-gpuDim[0])/2+cpuHaloWidth[0]+cpuForGpuHaloWidth[0];
        remoteHaloSpecsCpuForGpu[4].haloY = (localDim[1]-gpuDim[1])/2+cpuHaloWidth[3]+cpuForGpuHaloWidth[3];
        remoteHaloSpecsCpuForGpu[4].haloZ = (localDim[2]-gpuDim[2])/2+cpuHaloWidth[4];
        remoteHaloSpecsCpuForGpu[4].haloWidth = gpuDim[0]-cpuForGpuHaloWidth[0]-cpuForGpuHaloWidth[1];
        remoteHaloSpecsCpuForGpu[4].haloHeight = gpuDim[1]-cpuForGpuHaloWidth[2]-cpuForGpuHaloWidth[3];
        remoteHaloSpecsCpuForGpu[4].haloDepth = cpuForGpuHaloWidth[4];

        remoteHaloSpecsCpuForGpu[5].bufferWidth = tauschLocalDim[0];
        remoteHaloSpecsCpuForGpu[5].bufferHeight = tauschLocalDim[1];
        remoteHaloSpecsCpuForGpu[5].bufferDepth = tauschLocalDim[2];
        remoteHaloSpecsCpuForGpu[5].haloX = (localDim[0]-gpuDim[0])/2+cpuHaloWidth[0]+cpuForGpuHaloWidth[0];
        remoteHaloSpecsCpuForGpu[5].haloY = (localDim[1]-gpuDim[1])/2+cpuHaloWidth[3]+cpuForGpuHaloWidth[3];
        remoteHaloSpecsCpuForGpu[5].haloZ = (localDim[2]-gpuDim[2])/2+cpuHaloWidth[4]+gpuDim[2]-cpuForGpuHaloWidth[5];
        remoteHaloSpecsCpuForGpu[5].haloWidth = gpuDim[0]-cpuForGpuHaloWidth[0]-cpuForGpuHaloWidth[1];
        remoteHaloSpecsCpuForGpu[5].haloHeight = gpuDim[1]-cpuForGpuHaloWidth[2]-cpuForGpuHaloWidth[3];
        remoteHaloSpecsCpuForGpu[5].haloDepth = cpuForGpuHaloWidth[5];

        localHaloSpecsCpuForGpu[0].bufferWidth = tauschLocalDim[0];
        localHaloSpecsCpuForGpu[0].bufferHeight = tauschLocalDim[1];
        localHaloSpecsCpuForGpu[0].bufferDepth = tauschLocalDim[2];
        localHaloSpecsCpuForGpu[0].haloX = (localDim[0]-gpuDim[0])/2+cpuHaloWidth[0]-gpuHaloWidth[0];
        localHaloSpecsCpuForGpu[0].haloY = (localDim[1]-gpuDim[1])/2+cpuHaloWidth[3]-gpuHaloWidth[3];
        localHaloSpecsCpuForGpu[0].haloZ = (localDim[2]-gpuDim[2])/2+cpuHaloWidth[4]-gpuHaloWidth[4];
        localHaloSpecsCpuForGpu[0].haloWidth = gpuHaloWidth[0];
        localHaloSpecsCpuForGpu[0].haloHeight = gpuDim[1]+gpuHaloWidth[2]+gpuHaloWidth[3];
        localHaloSpecsCpuForGpu[0].haloDepth = gpuDim[2]+gpuHaloWidth[4]+gpuHaloWidth[5];

        localHaloSpecsCpuForGpu[1].bufferWidth = tauschLocalDim[0];
        localHaloSpecsCpuForGpu[1].bufferHeight = tauschLocalDim[1];
        localHaloSpecsCpuForGpu[1].bufferDepth = tauschLocalDim[2];
        localHaloSpecsCpuForGpu[1].haloX = (localDim[0]-gpuDim[0])/2+cpuHaloWidth[0]+gpuDim[0];
        localHaloSpecsCpuForGpu[1].haloY = (localDim[1]-gpuDim[1])/2+cpuHaloWidth[3]-gpuHaloWidth[3];
        localHaloSpecsCpuForGpu[1].haloZ = (localDim[2]-gpuDim[2])/2+cpuHaloWidth[4]-gpuHaloWidth[4];
        localHaloSpecsCpuForGpu[1].haloWidth = gpuHaloWidth[1];
        localHaloSpecsCpuForGpu[1].haloHeight = gpuDim[1]+gpuHaloWidth[2]+gpuHaloWidth[3];
        localHaloSpecsCpuForGpu[1].haloDepth = gpuDim[2]+gpuHaloWidth[4]+gpuHaloWidth[5];

        localHaloSpecsCpuForGpu[2].bufferWidth = tauschLocalDim[0];
        localHaloSpecsCpuForGpu[2].bufferHeight = tauschLocalDim[1];
        localHaloSpecsCpuForGpu[2].bufferDepth = tauschLocalDim[2];
        localHaloSpecsCpuForGpu[2].haloX = (localDim[0]-gpuDim[0])/2+cpuHaloWidth[0]-gpuHaloWidth[0];
        localHaloSpecsCpuForGpu[2].haloY = (localDim[1]-gpuDim[1])/2+cpuHaloWidth[3]+gpuDim[1];
        localHaloSpecsCpuForGpu[2].haloZ = (localDim[2]-gpuDim[2])/2+cpuHaloWidth[4]-gpuHaloWidth[4];
        localHaloSpecsCpuForGpu[2].haloWidth = gpuDim[0]+gpuHaloWidth[0]+gpuHaloWidth[1];
        localHaloSpecsCpuForGpu[2].haloHeight = gpuHaloWidth[2];
        localHaloSpecsCpuForGpu[2].haloDepth = gpuDim[2]+gpuHaloWidth[4]+gpuHaloWidth[5];

        localHaloSpecsCpuForGpu[3].bufferWidth = tauschLocalDim[0];
        localHaloSpecsCpuForGpu[3].bufferHeight = tauschLocalDim[1];
        localHaloSpecsCpuForGpu[3].bufferDepth = tauschLocalDim[2];
        localHaloSpecsCpuForGpu[3].haloX = (localDim[0]-gpuDim[0])/2+cpuHaloWidth[0]-gpuHaloWidth[0];
        localHaloSpecsCpuForGpu[3].haloY = (localDim[1]-gpuDim[1])/2+cpuHaloWidth[3]-gpuHaloWidth[3];
        localHaloSpecsCpuForGpu[3].haloZ = (localDim[2]-gpuDim[2])/2+cpuHaloWidth[4]-gpuHaloWidth[4];
        localHaloSpecsCpuForGpu[3].haloWidth = gpuDim[0]+gpuHaloWidth[0]+gpuHaloWidth[1];
        localHaloSpecsCpuForGpu[3].haloHeight = gpuHaloWidth[3];
        localHaloSpecsCpuForGpu[3].haloDepth = gpuDim[2]+gpuHaloWidth[4]+gpuHaloWidth[5];

        localHaloSpecsCpuForGpu[4].bufferWidth = tauschLocalDim[0];
        localHaloSpecsCpuForGpu[4].bufferHeight = tauschLocalDim[1];
        localHaloSpecsCpuForGpu[4].bufferDepth = tauschLocalDim[2];
        localHaloSpecsCpuForGpu[4].haloX = (localDim[0]-gpuDim[0])/2+cpuHaloWidth[0]-gpuHaloWidth[0];
        localHaloSpecsCpuForGpu[4].haloY = (localDim[1]-gpuDim[1])/2+cpuHaloWidth[3]-gpuHaloWidth[3];
        localHaloSpecsCpuForGpu[4].haloZ = (localDim[2]-gpuDim[2])/2+cpuHaloWidth[4]-gpuHaloWidth[4];
        localHaloSpecsCpuForGpu[4].haloWidth = gpuDim[0]+gpuHaloWidth[0]+gpuHaloWidth[1];
        localHaloSpecsCpuForGpu[4].haloHeight = gpuDim[1]+gpuHaloWidth[2]+gpuHaloWidth[3];
        localHaloSpecsCpuForGpu[4].haloDepth = gpuHaloWidth[4];

        localHaloSpecsCpuForGpu[5].bufferWidth = tauschLocalDim[0];
        localHaloSpecsCpuForGpu[5].bufferHeight = tauschLocalDim[1];
        localHaloSpecsCpuForGpu[5].bufferDepth = tauschLocalDim[2];
        localHaloSpecsCpuForGpu[5].haloX = (localDim[0]-gpuDim[0])/2+cpuHaloWidth[0]-gpuHaloWidth[0];
        localHaloSpecsCpuForGpu[5].haloY = (localDim[1]-gpuDim[1])/2+cpuHaloWidth[3]-gpuHaloWidth[3];
        localHaloSpecsCpuForGpu[5].haloZ = (localDim[2]-gpuDim[2])/2+cpuHaloWidth[4]+gpuDim[2];
        localHaloSpecsCpuForGpu[5].haloWidth = gpuDim[0]+gpuHaloWidth[0]+gpuHaloWidth[1];
        localHaloSpecsCpuForGpu[5].haloHeight = gpuDim[1]+gpuHaloWidth[2]+gpuHaloWidth[3];
        localHaloSpecsCpuForGpu[5].haloDepth = gpuHaloWidth[5];

        tausch->setLocalHaloInfo(TAUSCH_CPU|TAUSCH_WITHGPU, 6, localHaloSpecsCpuForGpu);
        tausch->setRemoteHaloInfo(TAUSCH_CPU|TAUSCH_WITHGPU, 6, remoteHaloSpecsCpuForGpu);

        remoteHaloSpecsGpu = new TauschHaloSpec[6];
        localHaloSpecsGpu = new TauschHaloSpec[6];

        localHaloSpecsGpu[0].bufferWidth = tauschGpuDim[0];
        localHaloSpecsGpu[0].bufferHeight = tauschGpuDim[1];
        localHaloSpecsGpu[0].bufferDepth = tauschGpuDim[2];
        localHaloSpecsGpu[0].haloX = gpuHaloWidth[0];
        localHaloSpecsGpu[0].haloY = gpuHaloWidth[3];
        localHaloSpecsGpu[0].haloZ = gpuHaloWidth[4];
        localHaloSpecsGpu[0].haloWidth = cpuForGpuHaloWidth[0];
        localHaloSpecsGpu[0].haloHeight = gpuDim[1];
        localHaloSpecsGpu[0].haloDepth = gpuDim[2];

        localHaloSpecsGpu[1].bufferWidth = tauschGpuDim[0];
        localHaloSpecsGpu[1].bufferHeight = tauschGpuDim[1];
        localHaloSpecsGpu[1].bufferDepth = tauschGpuDim[2];
        localHaloSpecsGpu[1].haloX = gpuDim[0]+gpuHaloWidth[0]-cpuForGpuHaloWidth[1];
        localHaloSpecsGpu[1].haloY = gpuHaloWidth[3];
        localHaloSpecsGpu[1].haloZ = gpuHaloWidth[4];
        localHaloSpecsGpu[1].haloWidth = cpuForGpuHaloWidth[1];
        localHaloSpecsGpu[1].haloHeight = gpuDim[1];
        localHaloSpecsGpu[1].haloDepth = gpuDim[2];

        localHaloSpecsGpu[2].bufferWidth = tauschGpuDim[0];
        localHaloSpecsGpu[2].bufferHeight = tauschGpuDim[1];
        localHaloSpecsGpu[2].bufferDepth = tauschGpuDim[2];
        localHaloSpecsGpu[2].haloX = gpuHaloWidth[0]+cpuForGpuHaloWidth[0];
        localHaloSpecsGpu[2].haloY = gpuDim[1]+gpuHaloWidth[3]-cpuForGpuHaloWidth[2];
        localHaloSpecsGpu[2].haloZ = gpuHaloWidth[4];
        localHaloSpecsGpu[2].haloWidth = gpuDim[0] - cpuForGpuHaloWidth[0] - cpuForGpuHaloWidth[1];
        localHaloSpecsGpu[2].haloHeight = cpuForGpuHaloWidth[2];
        localHaloSpecsGpu[2].haloDepth = gpuDim[2];

        localHaloSpecsGpu[3].bufferWidth = tauschGpuDim[0];
        localHaloSpecsGpu[3].bufferHeight = tauschGpuDim[1];
        localHaloSpecsGpu[3].bufferDepth = tauschGpuDim[2];
        localHaloSpecsGpu[3].haloX = gpuHaloWidth[0]+cpuForGpuHaloWidth[0];
        localHaloSpecsGpu[3].haloY = gpuHaloWidth[3];
        localHaloSpecsGpu[3].haloZ = gpuHaloWidth[4];
        localHaloSpecsGpu[3].haloWidth = gpuDim[0] - cpuForGpuHaloWidth[0] - cpuForGpuHaloWidth[1];
        localHaloSpecsGpu[3].haloHeight = cpuForGpuHaloWidth[3];
        localHaloSpecsGpu[3].haloDepth = gpuDim[2];

        localHaloSpecsGpu[4].bufferWidth = tauschGpuDim[0];
        localHaloSpecsGpu[4].bufferHeight = tauschGpuDim[1];
        localHaloSpecsGpu[4].bufferDepth = tauschGpuDim[2];
        localHaloSpecsGpu[4].haloX = gpuHaloWidth[0]+cpuForGpuHaloWidth[0];
        localHaloSpecsGpu[4].haloY = gpuHaloWidth[3]+cpuForGpuHaloWidth[3];
        localHaloSpecsGpu[4].haloZ = gpuHaloWidth[4];
        localHaloSpecsGpu[4].haloWidth = gpuDim[0] - cpuForGpuHaloWidth[0] - cpuForGpuHaloWidth[1];
        localHaloSpecsGpu[4].haloHeight = gpuDim[1] - cpuForGpuHaloWidth[2] - cpuForGpuHaloWidth[3];
        localHaloSpecsGpu[4].haloDepth = cpuForGpuHaloWidth[4];

        localHaloSpecsGpu[5].bufferWidth = tauschGpuDim[0];
        localHaloSpecsGpu[5].bufferHeight = tauschGpuDim[1];
        localHaloSpecsGpu[5].bufferDepth = tauschGpuDim[2];
        localHaloSpecsGpu[5].haloX = gpuHaloWidth[0]+cpuForGpuHaloWidth[0];
        localHaloSpecsGpu[5].haloY = gpuHaloWidth[3]+cpuForGpuHaloWidth[3];
        localHaloSpecsGpu[5].haloZ = gpuHaloWidth[4]+gpuDim[2]-cpuForGpuHaloWidth[5];
        localHaloSpecsGpu[5].haloWidth = gpuDim[0] - cpuForGpuHaloWidth[0] - cpuForGpuHaloWidth[1];
        localHaloSpecsGpu[5].haloHeight = gpuDim[1] - cpuForGpuHaloWidth[2] - cpuForGpuHaloWidth[3];
        localHaloSpecsGpu[5].haloDepth = cpuForGpuHaloWidth[5];

        remoteHaloSpecsGpu[0].bufferWidth = tauschGpuDim[0];
        remoteHaloSpecsGpu[0].bufferHeight = tauschGpuDim[1];
        remoteHaloSpecsGpu[0].bufferDepth = tauschGpuDim[2];
        remoteHaloSpecsGpu[0].haloX = 0;
        remoteHaloSpecsGpu[0].haloY = 0;
        remoteHaloSpecsGpu[0].haloZ = 0;
        remoteHaloSpecsGpu[0].haloWidth = gpuHaloWidth[0];
        remoteHaloSpecsGpu[0].haloHeight = gpuDim[1] + gpuHaloWidth[2]+gpuHaloWidth[3];
        remoteHaloSpecsGpu[0].haloDepth = gpuDim[2] + gpuHaloWidth[4]+gpuHaloWidth[5];

        remoteHaloSpecsGpu[1].bufferWidth = tauschGpuDim[0];
        remoteHaloSpecsGpu[1].bufferHeight = tauschGpuDim[2];
        remoteHaloSpecsGpu[1].bufferDepth = tauschGpuDim[2];
        remoteHaloSpecsGpu[1].haloX = gpuDim[0]+gpuHaloWidth[0];
        remoteHaloSpecsGpu[1].haloY = 0;
        remoteHaloSpecsGpu[1].haloZ = 0;
        remoteHaloSpecsGpu[1].haloWidth = gpuHaloWidth[1];
        remoteHaloSpecsGpu[1].haloHeight = gpuDim[1] + gpuHaloWidth[2]+gpuHaloWidth[3];
        remoteHaloSpecsGpu[1].haloDepth = gpuDim[2] + gpuHaloWidth[4]+gpuHaloWidth[5];

        remoteHaloSpecsGpu[2].bufferWidth = tauschGpuDim[0];
        remoteHaloSpecsGpu[2].bufferHeight = tauschGpuDim[1];
        remoteHaloSpecsGpu[2].bufferDepth = tauschGpuDim[2];
        remoteHaloSpecsGpu[2].haloX = 0;
        remoteHaloSpecsGpu[2].haloY = gpuDim[1]+gpuHaloWidth[3];
        remoteHaloSpecsGpu[2].haloZ = 0;
        remoteHaloSpecsGpu[2].haloWidth = gpuDim[0] + gpuHaloWidth[0]+gpuHaloWidth[1];
        remoteHaloSpecsGpu[2].haloHeight = gpuHaloWidth[2];
        remoteHaloSpecsGpu[2].haloDepth = gpuDim[2] + gpuHaloWidth[4]+gpuHaloWidth[5];

        remoteHaloSpecsGpu[3].bufferWidth = tauschGpuDim[0];
        remoteHaloSpecsGpu[3].bufferHeight = tauschGpuDim[1];
        remoteHaloSpecsGpu[3].bufferDepth = tauschGpuDim[2];
        remoteHaloSpecsGpu[3].haloX = 0;
        remoteHaloSpecsGpu[3].haloY = 0;
        remoteHaloSpecsGpu[3].haloZ = 0;
        remoteHaloSpecsGpu[3].haloWidth = gpuDim[0] + gpuHaloWidth[0]+gpuHaloWidth[1];
        remoteHaloSpecsGpu[3].haloHeight = gpuHaloWidth[3];
        remoteHaloSpecsGpu[3].haloDepth = gpuDim[2] + gpuHaloWidth[4]+gpuHaloWidth[5];

        remoteHaloSpecsGpu[4].bufferWidth = tauschGpuDim[0];
        remoteHaloSpecsGpu[4].bufferHeight = tauschGpuDim[1];
        remoteHaloSpecsGpu[4].bufferDepth = tauschGpuDim[2];
        remoteHaloSpecsGpu[4].haloX = 0;
        remoteHaloSpecsGpu[4].haloY = 0;
        remoteHaloSpecsGpu[4].haloZ = 0;
        remoteHaloSpecsGpu[4].haloWidth = gpuDim[0] + gpuHaloWidth[0]+gpuHaloWidth[1];
        remoteHaloSpecsGpu[4].haloHeight = gpuDim[1] + gpuHaloWidth[2]+gpuHaloWidth[3];
        remoteHaloSpecsGpu[4].haloDepth = gpuHaloWidth[4];

        remoteHaloSpecsGpu[5].bufferWidth = tauschGpuDim[0];
        remoteHaloSpecsGpu[5].bufferHeight = tauschGpuDim[1];
        remoteHaloSpecsGpu[5].bufferDepth = tauschGpuDim[2];
        remoteHaloSpecsGpu[5].haloX = 0;
        remoteHaloSpecsGpu[5].haloY = 0;
        remoteHaloSpecsGpu[5].haloZ = gpuDim[2]+gpuHaloWidth[4];
        remoteHaloSpecsGpu[5].haloWidth = gpuDim[0] + gpuHaloWidth[0]+gpuHaloWidth[1];
        remoteHaloSpecsGpu[5].haloHeight = gpuDim[1] + gpuHaloWidth[2]+gpuHaloWidth[3];
        remoteHaloSpecsGpu[5].haloDepth = gpuHaloWidth[5];

        tausch->setLocalHaloInfo(TAUSCH_GPU|TAUSCH_WITHCPU, 6, localHaloSpecsGpu);
        tausch->setRemoteHaloInfo(TAUSCH_GPU|TAUSCH_WITHCPU, 6, remoteHaloSpecsGpu);

    }

}

Sample::~Sample() {

    delete[] localHaloSpecs;
    delete[] remoteHaloSpecs;
    delete tausch;
    for(int b = 0; b < numBuffers; ++b)
        delete[] dat[b];
    delete[] dat;

}

void Sample::launchCPU() {

    if(hybrid) {

        int sendtagsCpu[6] = {0, 1, 2, 3, 4, 5};
        int recvtagsCpu[6] = {1, 0, 3, 2, 5, 4};
        int sendtagsGpu[6] = {0, 1, 2, 3, 4, 5};
        int recvtagsGpu[6] = {0, 1, 2, 3, 4, 5};

        for(int iter = 0; iter < loops; ++iter) {

            tausch->postAllReceives(TAUSCH_CPU|TAUSCH_WITHCPU, recvtagsCpu);
            tausch->postAllReceives(TAUSCH_GPU|TAUSCH_WITHCPU, recvtagsGpu);

            for(int hid = 0; hid < 3; ++hid) {

                // left/right

                for(int b = 0; b < numBuffers; ++b)
                    tausch->packSendBuffer(TAUSCH_CPU|TAUSCH_WITHCPU, 2*hid, b, dat[b]);
                tausch->send(TAUSCH_CPU|TAUSCH_WITHCPU, 2*hid, sendtagsCpu[2*hid]);

                for(int b = 0; b < numBuffers; ++b)
                    tausch->packSendBuffer(TAUSCH_CPU|TAUSCH_WITHCPU, 2*hid+1, b, dat[b]);
                tausch->send(TAUSCH_CPU|TAUSCH_WITHCPU, 2*hid+1, sendtagsCpu[2*hid+1]);

                for(int b = 0; b < numBuffers; ++b)
                    tausch->packSendBuffer(TAUSCH_CPU|TAUSCH_WITHGPU, 2*hid, b, dat[b]);
                tausch->send(TAUSCH_CPU|TAUSCH_WITHGPU, 2*hid, sendtagsGpu[2*hid]);

                for(int b = 0; b < numBuffers; ++b)
                    tausch->packSendBuffer(TAUSCH_CPU|TAUSCH_WITHGPU, 2*hid+1, b, dat[b]);
                tausch->send(TAUSCH_CPU|TAUSCH_WITHGPU, 2*hid+1, sendtagsGpu[2*hid+1]);

                tausch->recv(TAUSCH_CPU|TAUSCH_WITHCPU, 2*hid);
                for(int b = 0; b < numBuffers; ++b)
                    tausch->unpackRecvBuffer(TAUSCH_CPU|TAUSCH_WITHCPU, 2*hid, b, dat[b]);

                tausch->recv(TAUSCH_CPU|TAUSCH_WITHCPU, 2*hid+1);
                for(int b = 0; b < numBuffers; ++b)
                    tausch->unpackRecvBuffer(TAUSCH_CPU|TAUSCH_WITHCPU, 2*hid+1, b, dat[b]);

                tausch->recv(TAUSCH_CPU|TAUSCH_WITHGPU, 2*hid);
                for(int b = 0; b < numBuffers; ++b)
                    tausch->unpackRecvBuffer(TAUSCH_CPU|TAUSCH_WITHGPU, 2*hid, b, dat[b]);

                tausch->recv(TAUSCH_CPU|TAUSCH_WITHGPU, 2*hid+1);
                for(int b = 0; b < numBuffers; ++b)
                    tausch->unpackRecvBuffer(TAUSCH_CPU|TAUSCH_WITHGPU, 2*hid+1, b, dat[b]);

            }
        }

    } else {

        int sendtagsCpu[6] = {0, 1, 2, 3, 4, 5};
        int recvtagsCpu[6] = {1, 0, 3, 2, 5, 4};

        for(int iter = 0; iter < loops; ++iter) {

            tausch->postAllReceives(TAUSCH_CPU|TAUSCH_WITHCPU, recvtagsCpu);

            for(int hid = 0; hid < 3; ++hid) {

                // left/right

                for(int b = 0; b < numBuffers; ++b)
                    tausch->packSendBuffer(TAUSCH_CPU|TAUSCH_WITHCPU, 2*hid, b, dat[b]);
                tausch->send(TAUSCH_CPU|TAUSCH_WITHCPU, 2*hid, sendtagsCpu[2*hid]);

                for(int b = 0; b < numBuffers; ++b)
                    tausch->packSendBuffer(TAUSCH_CPU|TAUSCH_WITHCPU, 2*hid+1, b, dat[b]);
                tausch->send(TAUSCH_CPU|TAUSCH_WITHCPU, 2*hid+1, sendtagsCpu[2*hid+1]);

                tausch->recv(TAUSCH_CPU|TAUSCH_WITHCPU, 2*hid);
                for(int b = 0; b < numBuffers; ++b)
                    tausch->unpackRecvBuffer(TAUSCH_CPU|TAUSCH_WITHCPU, 2*hid, b, dat[b]);

                tausch->recv(TAUSCH_CPU|TAUSCH_WITHCPU, 2*hid+1);
                for(int b = 0; b < numBuffers; ++b)
                    tausch->unpackRecvBuffer(TAUSCH_CPU|TAUSCH_WITHCPU, 2*hid+1, b, dat[b]);

            }
        }

    }

}

void Sample::launchGPU() {

    for(int iter = 0; iter < loops; ++iter) {

        int sendtags[6] = {0, 1, 2, 3, 4, 5};
        int recvtags[6] = {0, 1, 2, 3, 4, 5};

        tausch->postAllReceives(TAUSCH_GPU|TAUSCH_WITHCPU, recvtags);

        for(int hid = 0; hid < 3; ++hid) {

            tausch->recv(TAUSCH_GPU|TAUSCH_WITHCPU, 2*hid);
            for(int b = 0; b < numBuffers; ++b)
                tausch->unpackRecvBuffer(TAUSCH_GPU|TAUSCH_WITHCPU, 2*hid, b, cl_gpudat[b]);

            tausch->recv(TAUSCH_GPU|TAUSCH_WITHCPU, 2*hid+1);
            for(int b = 0; b < numBuffers; ++b)
                tausch->unpackRecvBuffer(TAUSCH_GPU|TAUSCH_WITHCPU, 2*hid+1, b, cl_gpudat[b]);

            for(int b = 0; b < numBuffers; ++b)
                tausch->packSendBuffer(TAUSCH_GPU|TAUSCH_WITHCPU, 2*hid, b, cl_gpudat[b]);
            tausch->send(TAUSCH_GPU|TAUSCH_WITHCPU, 2*hid, sendtags[2*hid]);

            for(int b = 0; b < numBuffers; ++b)
                tausch->packSendBuffer(TAUSCH_GPU|TAUSCH_WITHCPU, 2*hid+1, b, cl_gpudat[b]);
            tausch->send(TAUSCH_GPU|TAUSCH_WITHCPU, 2*hid+1, sendtags[2*hid +1]);

        }

    }

    for(int b = 0; b < numBuffers; ++b) {
        int s = valuesPerPointPerBuffer[b]*(gpuDim[0] + gpuHaloWidth[0] + gpuHaloWidth[1])*(gpuDim[1] + gpuHaloWidth[2] + gpuHaloWidth[3])*(gpuDim[2] + gpuHaloWidth[4] + gpuHaloWidth[5]);
        cl::copy(tausch->getOpenCLQueue(), cl_gpudat[b], &gpudat[b][0], &gpudat[b][s]);
    }

}

void Sample::printCPU() {

    for(int z = 0; z < localDim[TAUSCH_Z]+cpuHaloWidth[4]+cpuHaloWidth[5]; ++z) {

        std::cout << std::endl << "z = " << z << std::endl;

        for(int j = localDim[TAUSCH_Y]+cpuHaloWidth[2]+cpuHaloWidth[3]-1; j >= 0; --j) {

            for(int b = 0; b < numBuffers; ++b) {

                for(int val = 0; val < valuesPerPointPerBuffer[b]; ++val) {
                    for(int i = 0; i < localDim[TAUSCH_X]+cpuHaloWidth[0]+cpuHaloWidth[1]; ++i)
                        std::cout << std::setw(4) << dat[b][valuesPerPointPerBuffer[b]*(
                                                            z*(localDim[TAUSCH_X]+cpuHaloWidth[0]+cpuHaloWidth[1])*
                                                            (localDim[TAUSCH_Y]+cpuHaloWidth[2]+cpuHaloWidth[3]) +
                                                          j*(localDim[TAUSCH_X]+cpuHaloWidth[0]+cpuHaloWidth[1]) + i) + val] << " ";
                    if(val != valuesPerPointPerBuffer[b]-1)
                        std::cout << "   ";
                }
                if(b != numBuffers-1)
                    std::cout << "          ";
            }
            std::cout << std::endl;
        }

    }

}

void Sample::printGPU() {

    for(int k = 0; k < gpuDim[2]+gpuHaloWidth[4]+gpuHaloWidth[5]; ++k) {

        std::cout << std::endl << "z = " << k << std::endl;

        for(int j = gpuDim[1]+gpuHaloWidth[2]+gpuHaloWidth[3]-1; j >= 0; --j) {
            for(int b = 0; b < numBuffers; ++b) {
                for(int val = 0; val < valuesPerPointPerBuffer[b]; ++val) {
                    for(int i = 0; i < gpuDim[0]+gpuHaloWidth[0]+gpuHaloWidth[1]; ++i)
                        std::cout << std::setw(4) << gpudat[b][valuesPerPointPerBuffer[b]*(k*(gpuDim[0]+gpuHaloWidth[0]+gpuHaloWidth[1])*
                                                                                             (gpuDim[1]+gpuHaloWidth[2]+gpuHaloWidth[3]) +
                                                                                           j*(gpuDim[0]+gpuHaloWidth[0]+gpuHaloWidth[1]) +
                                                                                           i) + val] << " ";
                    if(val != valuesPerPointPerBuffer[b]-1)
                        std::cout << "   ";
                }
                if(b != numBuffers-1)
                    std::cout << "          ";
            }
            std::cout << std::endl;
        }
    }

}
