#ifndef TAUSCHBASE_H
#define TAUSCHBASE_H

#ifdef TAUSCH_OPENCL
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#endif

typedef double real_t;
typedef int Edge;

/*!
 * Virtual API, allowing runtime choice of 2D or 3D version.
 */
class TauschBase {

};


#endif // TAUSCHBASE_H
