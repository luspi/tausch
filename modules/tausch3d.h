/*!
 * \file
 * \author  Lukas Spies <LSpies@illinois.edu>
 * \version 1.0
 */

#ifndef TAUSCH3D_H
#define TAUSCH3D_H

#include "tausch.h"
#include <mpi.h>
#include <iostream>

#ifndef TAUSCH_ENUM
#define TAUSCH_ENUM
namespace Tausch {
    /*!
     * These are the edges available for inter-MPI halo exchanges: LEFT, RIGHT, TOP, BOTTOM.
     */
    enum Edges { LEFT, RIGHT, TOP, BOTTOM };

    /*!
     * These are the two dimensions used, used for clarity as to which array entry is which dimension: X, Y.
     */
    enum Dimensions { X, Y, Z };

}
#endif // TAUSCH_ENUM

#endif // TAUSCH3D_H
