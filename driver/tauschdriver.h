#ifndef DRIVER_H
#define DRIVER_H

#include <random>
#include <mpi.h>
#include <tausch/tausch.h>

class TauschDriver {

public:
    explicit TauschDriver(size_t *localDim, int iterations, int *mpiNum);
    ~TauschDriver();

    void iterate();

private:
    size_t localDim[2];
    int mpiNum[2];
    int iterations;

    double deltaX, deltaY;

    int myRank, numProc;

    Tausch<double> *tausch;

    size_t left, right, top, bottom;
    size_t bottomleft, bottomright, topleft, topright;

    double *cpuData, *cpuStencil;

    TauschHaloSpec *remoteHaloSpecs;
    TauschHaloSpec *localHaloSpecs;

    void kernel(int startX, int startY, int endX, int endY);

};


#endif
