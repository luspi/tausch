/*!
 * \file
 * \author  Lukas Spies <LSpies@illinois.edu>
 * \version 1.0
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
 * \section intro Introduction
 *
 * Tausch is a library that provides a clean and efficient C/C++ API for halo exchange for structured grids. It supports halo exchange across the partition boundaries as specified by the user. It comes with an API for one dimension, Tausch1D (work in progress), for two dimensions, Tausch2D, and for three dimensions, Tausch3D (work in progress). The C API is a simple wrapper to the C++ API and also comes in two flavous, CTausch2D and CTausch3D.
 *
 * The interface is contained within a single header file \em tausch.h for both the C and C++ APIs. Nothing additional is required other than linking against the Tausch library. Both the C and C++ bindings are very similar. The underlying C++ API is documented here in detail, the C API works equivalently, with the object being called CTausch and any method having the prefix 'tausch_'.
 *
 * \em Note: Tausch requires C++11 support to work!
 *
 */

#ifndef TAUSCH_H
#define TAUSCH_H

#include "modules/tausch1d.h"
#include "modules/tausch2d.h"
#include "modules/tausch3d.h"

#include "modules/ctausch1d.h"
#include "modules/ctausch2d.h"
#include "modules/ctausch3d.h"

#endif // TAUSCH_H
