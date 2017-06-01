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
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of
 * the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details at
 * https://www.gnu.org/copyleft/gpl.html
 *
 * \mainpage
 *
 * \section intro Introduction
 *
 * Tausch (pronounced:\htmlonly [ta&upsilon;&#x0283;\endhtmlonly\latexonly [ta\textipa{uS}\endlatexonly]) is a library that provides a clean and
 * efficient C/C++ API for halo exchange for structured grids. It supports halo exchange across the partition boundaries as specified by the user.
 * It comes with an API for one dimension, Tausch1D, for two dimensions, Tausch2D, and for three dimensions, Tausch3D. The C API is a simple wrapper
 * to the C++ API, taking advantage of the polymorphism of C++. Thus, for all three versions of Tausch there
 * is one C wrapper for each datatype (differentiated by a suffix), e.g., \link ctauschdouble.h CTauschDouble\endlink for the double datatype.
 *
 * The interface is contained within a single header file \em tausch.h for both the C and C++ APIs. Nothing additional is required other than linking
 * against the %Tausch library. Both the C and C++ bindings are very similar. Both the underlying C++ API, and the C wrapper API are documented here
 * in detail.
 *
 * \em Note: %Tausch requires C++11 support to work!
 *
 * \section assumptions Assumptions
 * Only very few assumptions are made regarding how the overall domain is partitioned amongst MPI ranks and regarding how the GPU handles a
 * subpartition of the overall partition. The assumptions that are done can be put into two groups:
 * -# <b>Data layout</b>: The remote halo for a partition is part of the same buffer the data owned by that partition, i.e., there is one big buffer
 * containing both. The data is expected to be stored first along the x dimension, then along the y dimension, and finally along the z dimension,
 * depending on how many dimensions are required. The halo along each edge is expected to be the same all along the edge, i.e., in two dimensions, for
 * example, the remote halo along the left edge has the same width all along the left edge. However, the halo does <i>not</i> have to be the same for
 * all the edges, each edge is treated seperately.
 * -# <b>GPU fully inside</b>: Any GPU partition (if available) lies fully inside the MPI partition and is not involved with any inter-MPI
 * communication. This restriction is planned to be removed in the future.
 *
 * \section terminology Terminology
 * For clarification, there are two types of halos that is referred to in the documentation and also the API itself:
 * - <b>Local halo</b>: the region of the partition owned by the current processor that is requested by other processors as halo data
 * - <b>Remote halo</b>: the halo required by the current processor for computations that lives on other processors
 *
 * \section possible What is possible
 * Due to making only very few assumption, %Tausch is very flexible and can be used for many different scenarios:
 * - If there are multiple buffers covering a domain, then their halos can all be sent in one combined message for each halo region, assuming they all
 * use the same discretisation over the whole domain. They each are packed consecutively, one after the other. I.e., as soon as one becomes available
 * it can be packed, preserving the order in which they are packed.
 * - If there is one buffer that stores multiple values for each point consecutively (e.g., when storing a stencil) then they can also get sent as one
 * grouped message, assuming that each point has the same number of values stored.
 * - The use of templates allows %Tausch to be used for data of different types. Tausch supports most of the common C/C++ datatypes:
 * char, char16_t, char32_t, wchar_t, signed char, short int, int, long, long long, unsigned char, unsigned short int, unsigned int, unsigned long,
 * unsigned long long, float, double, long double, bool.
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
 * 3. The receiving MPI rank (if region is required by other MPI rank) or the sending MPI rank (if region lives on other MPI rank is required by this
 * one)
 *
 * This set of halo regions needs to be specified twice: For the local halo regions and for the remote halo regions. Once this information is
 * specified, performing a halo exchange is very simple. Generally, the following four steps are necessary, though the first two (sending off local
 * halo data) or the last two (receiving rmeote halo data) could be optional should they not be required.
 * 1. Pack the a provided data buffer for a specific local halo region into a dedicated MPI send buffer.
 * 2. Send off the MPI send buffer for a specified local halo region.
 * 3. Received an incoming message for a specific remote halo region.
 * 4. Unpack a received message for a specific remote halo region into a specified data buffer.
 *
 * \section code Code snippet
 * Here you can find a short code that uses %Tausch for a halo exchange in two dimensions across a structured grid. For simplicity, we will only
 * perform a halo exchange to the right, across the right edge. If the right edge is along the domain boundary it wraps around to the opposite end
 * again (periodic boundary conditions). It shows how to use %Tausch with two buffers over the same domain. At the end it outputs the required
 * runtime.
 *
 * \includelineno samplecode.cpp
 *
 * This code can be compiled with: `mpic++ samplecode.cpp -ltausch -O3 -o sample`
 *
 */

#ifndef TAUSCH_H
#define TAUSCH_H

/*!
 * An enum to choose at runtime which version of Tausch to use: 1D, 2D or 3D. This enum is only used for the C API!
 */
enum TauschVersion {
    TAUSCH_1D,
    TAUSCH_2D,
    TAUSCH_3D
};




#include "modules/tausch1d.h"
#include "modules/tausch2d.h"
#include "modules/tausch3d.h"

#include "modules/ctausch/ctauschdouble.h"
#include "modules/ctausch/ctauschfloat.h"
#include "modules/ctausch/ctauschint.h"
#include "modules/ctausch/ctauschunsignedint.h"
#include "modules/ctausch/ctauschbool.h"

#endif // TAUSCH_H
