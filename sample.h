#ifndef SAMPLE_H
#define SAMPLE_H

#include "tausch.h"
#include <fstream>
#include <cmath>

class Sample {

public:
    explicit Sample();

private:
    int dim;
    int mpi_rank, mpi_size;

    void EveryoneOutput(const std::string &inMessage);

};

#endif
