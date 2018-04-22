#include <iostream>
#include <cmath>
#include <chrono>
#include "tauschdriver.h"

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    int myRank, numProc;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProc);

    int iterations = 10;
    size_t localDim[2] = {100,100};
    int mpiNum[2] = {(int)std::sqrt(numProc), (int)std::sqrt(numProc)};
    int averageOfHowManyRuns = 10;

    for(int i = 1; i < argc; ++i) {

        if(std::string(argv[i]) == "-iter" && i+1 < argc)
            iterations = atoi(argv[++i]);
        else if(std::string(argv[i]) == "-x" && i+1 < argc)
            localDim[0] = atoll(argv[++i]);
        else if(std::string(argv[i]) == "-y" && i+1 < argc)
            localDim[1] = atoi(argv[++i]);
        else if(std::string(argv[i]) == "-xy" && i+1 < argc) {
            localDim[0] = atoi(argv[++i]);
            localDim[1] = localDim[0];
        } else if(std::string(argv[i]) == "-mpix" && i+1 < argc)
            mpiNum[0] = atoi(argv[++i]);
        else if(std::string(argv[i]) == "-mpiy" && i+1 < argc)
            mpiNum[1] = atoi(argv[++i]);
        else if(std::string(argv[i]) == "-time" && i+1 < argc)
            averageOfHowManyRuns = atoi(argv[++i]);
    }

    if(myRank == 0) {

        std::cout << std::endl
                  << "*********************************" << std::endl
                  << " TauschDriver Configuration" << std::endl
                  << std::endl
                  << "  iterations           = " << iterations << std::endl
                  << "  localDim             = " << localDim[0] << "/" << localDim[1] << std::endl
                  << "  mpiNum               = " << mpiNum[0] << "/" << mpiNum[1] << std::endl
                  << "  averageOfHowManyRuns = " << averageOfHowManyRuns << std::endl
                  << std::endl
                  << "*********************************" << std::endl
                  << std::endl;

    }

    if(mpiNum[0]*mpiNum[1] != numProc) {
        std::cout << "Invalid number of ranks, requested: " << numProc << " - specified for use: " << mpiNum[0]*mpiNum[1] << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    TauschDriver driver(localDim, iterations, mpiNum);

    double *allt = new double[averageOfHowManyRuns]{};

    for(int run = 0; run < averageOfHowManyRuns; ++run) {

        MPI_Barrier(MPI_COMM_WORLD);
        auto t_start = std::chrono::steady_clock::now();

        driver.iterate();

        MPI_Barrier(MPI_COMM_WORLD);
        auto t_end = std::chrono::steady_clock::now();
        double t = std::chrono::duration<double, std::milli>(t_end-t_start).count();

        if(myRank == 0)
            std::cout << " Test #" << run << " took " << t << " ms" << std::endl;

        allt[run] = t;

    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(myRank == 0) {

        double total = std::accumulate(&allt[0], &allt[averageOfHowManyRuns], 0);

        std::cout << std::endl
                  << " >> Average time is: " << total/(double)averageOfHowManyRuns << " ms" << std::endl
                  << std::endl;

    }

    MPI_Finalize();

    return 0;

}
