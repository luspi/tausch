#ifndef SAMPLE_H
#define SAMPLE_H

#include <mpi.h>
#include <tausch/tausch.h>
#include <iomanip>

class Sample {

public:
    explicit Sample(size_t *localDim, size_t loops, size_t *cpuHaloWidth, size_t *mpiNum);
    ~Sample();

    void launchCPU();

    void print();

private:
    size_t localDim[3];
    size_t loops;
    size_t cpuHaloWidth[6];
    size_t mpiNum[3];

    Tausch<double> *tausch;

    TauschHaloSpec *localHaloSpecs;
    TauschHaloSpec *remoteHaloSpecs;
    double *dat1, *dat2;
    size_t numBuffers;
    size_t valuesPerPoint[2];

    size_t left, right, top, bottom, front, back;

};

#endif // SAMPLE_H
