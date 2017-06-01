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
    size_t localDim[2];
    size_t loops;
    size_t cpuHaloWidth[4];
    size_t mpiNum[2];

    Tausch<double> *tausch;

    size_t **localHaloSpecs;
    size_t **remoteHaloSpecs;
    double *dat1, *dat2;
    size_t numBuffers;
    size_t valuesPerPoint;

    size_t left, right, top, bottom;

};

#endif // SAMPLE_H
