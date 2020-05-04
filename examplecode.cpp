#include "tausch.h"
#include <mpi.h>

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    // Get the current rank
    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    // We send from the first buffer into the second buffer
    double *buffer1 = new double[5]{};
    double *buffer2 = new double[10]{};

    // The first buffer is filled with values from 1 to 5.
    for(int i = 0; i < 5; ++i)
        buffer1[i] = i+1;

    // Output buffer1
    if(myRank == 0) {
        std::cout << " Input buffer: ";
        for(int i = 0; i < 5; ++i)
            std::cout << buffer1[i] << " ";
        std::cout << std::endl;
    }

    // Halo indices: Sending all 5 values from buffer1, filling them into second buffer spaced apart
    std::vector<int> sendIndices = {0,1,2,3,4};
    std::vector<int> recvIndices = {0,2,4,6,8};

    // Create new Tausch object
    Tausch *tausch = new Tausch();

    // Set halo information
    tausch->addSendHaloInfo(sendIndices, sizeof(double));
    tausch->addRecvHaloInfo(recvIndices, sizeof(double));

    // pack, send, recv, and unpack data
    tausch->packSendBuffer(0, 0, buffer1);
    tausch->send(0, 0, myRank);
    tausch->recv(0, 0, myRank);
    tausch->unpackRecvBuffer(0, 0, buffer2);

    // Output buffer2
    if(myRank == 0) {
        std::cout << "Output buffer: ";
        for(int i = 0; i < 10; ++i)
            std::cout << buffer2[i] << " ";
        std::cout << std::endl;
    }

    // Done!
    MPI_Finalize();
    return 0;

}
