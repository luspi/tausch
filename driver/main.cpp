#include <iostream>
#include <cmath>
#include <chrono>
#include "tauschdriver.h"

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    int myRank, numProc;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProc);

    // Default values
    int iterations = 10;
    int localDim[2] = {100,100};
    int mpiNum[2] = {(int)std::sqrt(numProc), (int)std::sqrt(numProc)};
    int averageOfHowManyRuns = 10;

    // check for command line options
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

    // run at least 3 times (needed as largest and smallest times are always dropped)
    if(averageOfHowManyRuns < 3)
        averageOfHowManyRuns = 3;

    // Output configuration
    if(myRank == 0) {

        std::cout << std::endl
                  << "*********************************" << std::endl
                  << " TauschDriver Configuration" << std::endl
                  << std::endl
                  << "  iterations           = " << iterations << std::endl
                  << "  localDim             = " << localDim[0] << "/" << localDim[1] << std::endl
                  << "  mpiNum               = " << mpiNum[0] << "x" << mpiNum[1] << std::endl
                  << "  averageOfHowManyRuns = " << averageOfHowManyRuns << std::endl
                  << std::endl
                  << "*********************************" << std::endl
                  << std::endl;

    }

    // If something is off, display warning and quit
    if(mpiNum[0]*mpiNum[1] != numProc) {
        std::cout << "Invalid number of ranks, requested: " << numProc << " - specified for use: " << mpiNum[0]*mpiNum[1] << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Driver object
    TauschDriver driver(localDim, iterations, mpiNum);

    // Handles the timings (all times and the largest/smallest one)
    double *allt = new double[averageOfHowManyRuns]{};
    double mint = 999999999999, maxt = 0;
    int mintpos = 0, maxtpos = 0;

    // Do the requested number of timing runs
    for(int run = 0; run < averageOfHowManyRuns; ++run) {

        // All ranks start at the same time
        MPI_Barrier(MPI_COMM_WORLD);

        auto t_start = std::chrono::steady_clock::now();

        // Perform all iterations (applies stencil and exchanges halo data)
        driver.iterate();

        // Wait for all ranks to end
        MPI_Barrier(MPI_COMM_WORLD);
        auto t_end = std::chrono::steady_clock::now();

        // Calculate the total time needed
        double t = std::chrono::duration<double, std::milli>(t_end-t_start).count();

        // Output timing for each timing run
        if(myRank == 0)
            std::cout << " Run #" << run << " took " << t << " ms" << std::endl;

        // Store time
        allt[run] = t;
        // smallest time?
        if(t < mint) {
            mint = t;
            mintpos = run;
        }
        // largest time?
        if(t > maxt) {
            maxt = t;
            maxtpos = run;
        }

    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Output result
    if(myRank == 0) {

        // Find total sum of all timings (without smallest and largest time)
        double total = 0;
        for(int i = 0; i < averageOfHowManyRuns; ++i) {
            if(i == mintpos || i == maxtpos)
                continue;
            total += allt[i];
        }

        // Display average time
        std::cout << std::endl
                  << " >> Average time is: " << total/(double)(averageOfHowManyRuns-2) << " ms" << std::endl
                  << std::endl;

    }

    MPI_Finalize();

    return 0;

}
