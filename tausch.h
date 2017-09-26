/*!
 * \file
 * \author  Lukas Spies <LSpies@illinois.edu>
 * \version 1.0
 *
 * \brief
 *  Header to be used by user.
 *
 * Header providing access to any version of API. This is the one the user is strongly encouraged to include in any project instead of the individual
 * module headers!
 *
 * \section LICENSE
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2017, Lukas Spies
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *
 * \mainpage
 *
 * \section intro Introduction
 *
 * Tausch (pronounced:\htmlonly [ta&upsilon;&#x0283;\endhtmlonly\latexonly [ta\textipa{uS}\endlatexonly]) is a library that provides a clean and
 * efficient C/C++ API for halo exchange for structured grids. It supports halo exchange across the partition boundaries as specified by the user.
 * It comes with an API for one dimension, Tausch1D, for two dimensions, Tausch2D, and for three dimensions, Tausch3D. The C API is a simple wrapper
 * to the C++ API, taking advantage of the polymorphism of C++. Thus, for all three versions of Tausch there
 * is one C wrapper for the most common data types (differentiated by a suffix); see \link ctauschdouble.h CTauschDouble\endlink for the double data
 * type.
 *
 * The interface is contained within a single header file \em tausch.h for both the C and C++ APIs. Nothing additional is required other than linking
 * against the %Tausch library. Both the C and C++ bindings are very similar. The underlying C++ API, and the C wrapper API are documented here
 * in detail. For the C API, only the version for the double data type is documented, the other data types work equivalently, only with different
 * suffix.
 *
 * \em Note: %Tausch requires at least C++11 support to work!
 *
 * \section terminology Terminology
 * For clarification, there are two types of halos that is referred to in the documentation and also the API itself:
 * - <b>Local halo</b>: the region of the partition owned by the current processor that is requested by other processors as halo data
 * - <b>Remote halo</b>: the halo required by the current processor for computations that lives on other processors
 *
 * \section assumptions Assumptions
 * Tausch makes almost no assumptions about the data and the overall domain. How the domain is partitioned or where the GPU lies, Tausch does not care
 * about these specifics. There are only two assumptions done by Tausch, both very basic:
 * -# <b>Data layout</b>: The data is expected to be stored first along the x, then along the y, and finally along the z dimension. Within a specified
 * buffer dimension (specified within each halo specification), the data is expected to be stored as one contiguous array, and also containing the
 * halo data.
 * -# <b>GPU fully inside</b>: Currently, Tausch only supports communication between the CPU and GPU when both live on the same MPI rank. This
 * restriction is planned to be removed in the future.
 *
 * \section possible What is possible
 * Due to making only very few assumption, %Tausch is very flexible and can be used for many different scenarios:
 * - If there are multiple buffers covering a domain, then their halos can all be sent in one combined message for each halo region. The buffers do
 * not even need to have the same dimensions, but must follow the same underlying discretisation. The order in which they are packed can be controlled
 * using the buffer id, i.e., any order is possible.
 * - If there is one buffer that stores multiple values for each point consecutively (e.g., when storing a stencil) then they can also get sent as one
 * grouped message, assuming that each point has the same number of values stored.
 * - The use of templates allows %Tausch to be used for data of different types. Tausch supports most of the common C/C++ datatypes, with the C API
 * offering support for the most common of these.
 * - A common base class amongst all three versions (1D, 2D, and 3D) containing virtual pointers to all functions in the API allows the user to choose
 * at runtime which version to use. Using virtual function pointers does not appear to cause any slowdown in the case of %Tausch.
 * - The buffers do not have to be the same throughout the lifetime of any %Tausch object. When packing a buffer, %Tausch requires a pointer to the
 * data passed on, i.e., the buffer can be changed should that be required, as long as the discretisation is kept the same.
 *
 * \section overview High level API overview
 * You can find the details of the API in the documentation of each individual function, but here is a high level overview of how the API works:
 *
 * Tausch expects to be told for each partition how big the partition is, how wide the different halos are, and which part of the partition is needed
 * by another MPI rank and which parts of the halo are filed by which other MPI rank. Thus, for each of the different halo regions it needs to be told
 * the following information:
 * 1. The x/y/z coordinates of start of the halo region
 * 2. The width/height/depth of the halo region
 * 2. The width/height/depth of the underlying buffer - the buffer does not necessarily need to cover the whole domain
 * 3. The receiving MPI rank (if region is required by other MPI rank) or the sending MPI rank (if region lives on other MPI rank is required by this
 * one)
 *
 * This set of halo regions needs to be specified twice: For the local halo regions and for the remote halo regions. Once this information is
 * specified, performing a halo exchange is very simple. Generally, the following four steps are necessary, though the first two (sending off local
 * halo data) or the last two (receiving rmeote halo data) could be optional should they not be required.
 * 1. Pack the a provided data buffer for a specific local halo region into a dedicated send buffer.
 * 2. Send off the send buffer for a specified local halo region.
 * 3. Received an incoming message for a specific remote halo region.
 * 4. Unpack a received message for a specific remote halo region into a specified data buffer.
 *
 * The following code snippet demonstrates the steps detailed above. It omits all the details around for readability. For a full compilable example,
 * see below.
 *
 * \include codesnippet.cpp
 *
 * \section code Code snippet
 * This is a short but compilable example code of how to use %Tausch for a halo exchange in two dimensions across a structured grid. For simplicity,
 * we will only perform a halo exchange to the right, across the right edge. If the right edge is along the domain boundary it wraps around to the
 * opposite end again (periodic boundary conditions). It shows how to use %Tausch with two buffers over the same domain. At the end it outputs the
 * required runtime.
 *
 * \includelineno samplecode.cpp
 *
 * This code can be compiled with: `mpic++ samplecode.cpp -ltausch -std=c++11 -latomic -O3 -o sample`
 *
 * \section license License
 *
 * \code{.unparsed}
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2017, Lukas Spies
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * \endcode
 *
 */

#ifndef TAUSCH_H
#define TAUSCH_H

#include "modules/tausch1d.h"
#include "modules/tausch2d.h"
#include "modules/tausch3d.h"

#include "modules/ctausch/ctauschdouble.h"
#include "modules/ctausch/ctauschfloat.h"
#include "modules/ctausch/ctauschint.h"
#include "modules/ctausch/ctauschunsignedint.h"

#endif // TAUSCH_H
