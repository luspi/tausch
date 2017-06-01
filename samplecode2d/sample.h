#ifndef SAMPLE_H
#define SAMPLE_H

#include <mpi.h>
#include <tausch/tausch.h>
#include <iomanip>

class Sample {

public:
    explicit Sample(int *localDim, int loops, int *cpuHaloWidth, int *mpiNum);
    ~Sample();

    void launchCPU();

    void print();

private:
    int localDim[2];
    int loops;
    int cpuHaloWidth[4];
    int mpiNum[2];

    Tausch<double> *tausch;

    int **localHaloSpecs;
    int **remoteHaloSpecs;
    double *dat1, *dat2;
    int numBuffers;
    int valuesPerPoint;

    int left, right, top, bottom;

};

#endif // SAMPLE_H
