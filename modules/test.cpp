#include "tausch.h"
#include <cstdlib>
#include <mpi.h>
#include <iomanip>
#include <vector>
#include <cmath>
#include <chrono>

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int mpiX = std::sqrt(size);
    int mpiY = mpiX;
    if(mpiX*mpiY != size) {
        mpiX = size;
        mpiY = 1;
    }

    size_t xy = 1000;
    size_t maxiter = 1000;

    for(int i = 0; i < argc; ++i) {
        if(argv[i] == std::string("-xy") && i < argc-1)
            xy = size_t(atoi(argv[++i]));
        else if(argv[i] == std::string("-maxiter") && i < argc-1)
            maxiter = size_t(atoi(argv[++i]));
    }

    if(rank == 0) {
        std::cout << "  CONFIGURATION" << std::endl;
        std::cout << "> xy = " << xy << std::endl;
        std::cout << "> maxiter = " << maxiter << std::endl << std::endl << std::endl;
    }

    double *buf = new double[(xy+2)*(xy+2)]{};

    for(int y = 0; y < xy; ++y)
        for(int x = 0; x < xy; ++x)
            buf[(y+1)*(xy+2) + x+1] = rank*100 + y*xy + x+1;

    Tausch<double> *tausch = new Tausch<double>(MPI_DOUBLE);

    std::vector<size_t> sendindices, recvindices;

    // LEFT
    for(int i = 0; i < xy+2; ++i) {
        sendindices.push_back(1+i*(xy+2));
        recvindices.push_back(i*(xy+2));
    }
    tausch->addLocalHaloInfo(sendindices);
    tausch->addRemoteHaloInfo(recvindices);

    sendindices.clear();
    recvindices.clear();

    // RIGHT
    for(int i = 0; i < xy+2; ++i) {
        sendindices.push_back((xy+2)-2 + i*(xy+2));
        recvindices.push_back((xy+2)-1 + i*(xy+2));
    }
    tausch->addLocalHaloInfo(sendindices);
    tausch->addRemoteHaloInfo(recvindices);

    sendindices.clear();
    recvindices.clear();

    // TOP
    for(int i = 0; i < xy+2; ++i) {
        sendindices.push_back((xy+2)*xy + i);
        recvindices.push_back((xy+2)*(xy+1) + i);
    }
    tausch->addLocalHaloInfo(sendindices);
    tausch->addRemoteHaloInfo(recvindices);

    sendindices.clear();
    recvindices.clear();

    // BOTTOM
    for(int i = 0; i < xy+2; ++i) {
        sendindices.push_back(xy+2 + i);
        recvindices.push_back(i);
    }
    tausch->addLocalHaloInfo(sendindices);
    tausch->addRemoteHaloInfo(recvindices);

    int left = rank-1, right = rank+1, top = rank+mpiX, bottom = rank-mpiX;

    // left edge
    if(rank%mpiX == 0)
        left = rank+mpiX-1;
    // right edge
    if((rank+1)%mpiX == 0)
        right = rank-mpiX+1;
    // top edge
    if(rank > size-mpiX-1)
        top = rank%mpiX;
    // bottom edge
    if(rank < mpiX)
        bottom = size-mpiX+rank;


    MPI_Barrier(MPI_COMM_WORLD);
    auto t1 = std::chrono::steady_clock::now();

    for(size_t iter = 0; iter < maxiter; ++iter) {

        tausch->packAndSend(0, buf, 0, left);
        tausch->packAndSend(1, buf, 1, right);
        tausch->recvAndUnpack(0, buf, 1, left);
        tausch->recvAndUnpack(1, buf, 0, right);

        tausch->packAndSend(2, buf, 2, top);
        tausch->packAndSend(3, buf, 3, bottom);
        tausch->recvAndUnpack(2, buf, 3, top);
        tausch->recvAndUnpack(3, buf, 2, bottom);

    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t2 = std::chrono::steady_clock::now();

    if(rank == 0)
        std::cout << "Time taken: " << std::chrono::duration<double, std::milli>(t2-t1).count() << " ms" << std::endl;


//     if(rank == 0) {
//         std::cout << std::endl << std::endl << "AFTER" << std::endl;
//         for(int y = xy+1; y >= 0; --y) {
//             for(int x = 0; x < xy+2; ++x)
//                 std::cout << std::setw(5) << buf[y*(xy+2) + x];
//             std::cout << std::endl;
//         }
//     }

    delete buf;

    MPI_Finalize();

    return 0;

}
