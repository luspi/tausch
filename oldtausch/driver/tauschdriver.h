#ifndef DRIVER_H
#define DRIVER_H

#include <random>
#include <mpi.h>
#include "../tausch.h"

class TauschDriver {

public:
    explicit TauschDriver(int *localDim, int iterations, int *mpiNum);
    ~TauschDriver();

    void iterate();

private:
    int myRank, numProc;

    int localDim[2];
    int iterations;

    int mpiNum[2];
    int left, right, top, bottom;

    Tausch<double> *tausch;

    int *localHaloIndices;
    int *remoteHaloIndices;

    double *cpuData, *cpuStencil;

    void kernel(int startX, int startY, int endX, int endY);

};


#endif
