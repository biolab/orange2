// Magic Software, Inc.
// http://www.magic-software.com
// http://www.wild-magic.com
// Copyright (c) 2004.  All Rights Reserved
//
// The Wild Magic Library (WML) source code is supplied under the terms of
// the license agreement http://www.magic-software.com/License/WildMagic.pdf
// and may not be copied or disclosed except in accordance with the terms of
// that agreement.

#ifndef WMLSYSTEM_H
#define WMLSYSTEM_H

//----------------------------------------------------------------------------
// Microsoft Windows
//----------------------------------------------------------------------------
#if defined(WIN32)

// for a DLL library
#if defined(WML_DLL_EXPORT)
#define WML_ITEM __declspec(dllexport)

// for a client of the DLL library
#elif defined(WML_DLL_IMPORT)
#define WML_ITEM __declspec(dllimport)

// for a static library
#else
#define WML_ITEM

#endif

#if defined(_MSC_VER)

// Microsoft Visual C++ specific pragmas.  MSVC6 appears to be version 1200
// and MSVC7 appears to be version 1300.
#if _MSC_VER < 1300
#define WML_USING_VC6
#else
#define WML_USING_VC7
#endif

#if defined(WML_USING_VC6)

// Disable the warning about truncating the debug names to 255 characters.
// This warning shows up often with STL code in MSVC6, but not MSVC7.
#pragma warning( disable : 4786 )

// This warning is disabled because MSVC6 warns about not finding
// implementations for the pure virtual functions that occur in the template
// classes 'template <class Real>' when explicity instantiating the classe.
// NOTE:  If you create your own template classes that will be explicitly
// instantiated, you should re-enable the warning to make sure that in fact
// all your member data and functions have been defined and implemented.
#pragma warning( disable : 4661 )

#endif

// TO DO.  What does this warning mean?
// warning C4251:  class 'std::vector<_Ty,_Ax>' needs to have dll-interface
//   to be used by clients of class 'foobar'
#pragma warning( disable : 4251 )

#endif

// Specialized instantiation of static members in template classes before or
// after the class itself is instantiated is not a problem with Visual Studio
// .NET 2003 (VC 7.1), but VC 6 likes the specialized instantiation to occur
// after the class instantiation.
// #define WML_INSTANTIATE_BEFORE

// common standard library headers
#include <cassert>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <sys/stat.h>
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// Macintosh OS X
//----------------------------------------------------------------------------
#elif defined(__APPLE__)

#define WML_BIG_ENDIAN

// Macro used for Microsoft Windows systems to support dynamic link libraries.
// Not needed for the Macintosh.
#define WML_ITEM

// g++ wants specialized template instantiations to occur after the
// explicit class instantiations.  CodeWarrior wants them to occur
// before.
#ifdef __MWERKS__
#define WML_INSTANTIATE_BEFORE
#endif

#include <cassert>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <sys/stat.h>
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// SGI IRIX
//----------------------------------------------------------------------------
#elif defined(SGI_IRIX)

// Macro used for Microsoft Windows systems to support dynamic link libraries.
// Not needed for Linux.
#define WML_ITEM

// Specialized instantiation of static members in template classes must occur
// before the class itself is instantiated.
#define WML_INSTANTIATE_BEFORE

// common standard library headers
#ifdef WML_IRIX_OLD_STYLE_HEADERS
#include <assert.h>
#include <ctype.h>
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#else
#include <cassert>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <sys/stat.h>
#endif
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// HP-UX
//----------------------------------------------------------------------------
#elif defined(HP_UX)

// Macro used for Microsoft Windows systems to support dynamic link libraries.
// Not needed for Linux.
#define WML_ITEM

// Specialized instantiation of static members in template classes must occur
// before the class itself is instantiated.
#define WML_INSTANTIATE_BEFORE

// common standard library headers
#include <cassert>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <sys/stat.h>
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// Sun SOLARIS
//----------------------------------------------------------------------------
#elif defined(SUN_SOLARIS)

// Macro used for Microsoft Windows systems to support dynamic link libraries.
// Not needed for Linux.
#define WML_ITEM

// The compilation on Solaris was successful before the addition of
// WML_INSTANTIATE_BEFORE.  If a compiler on the Sun likes the instantiation
// before, uncomment this define.
// #define WML_INSTANTIATE_BEFORE

// common standard library headers
#include <cassert>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <sys/stat.h>
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// Linux
//----------------------------------------------------------------------------
#else

// Macro used for Microsoft Windows systems to support dynamic link libraries.
// Not needed for Linux.
#define WML_ITEM

// Linux on a PC. Red Hat 8.x g++ has problems with specialized instantiation
// of static members in template classes *before* the class itself is
// explicitly instantiated.  The problem is not consistent; for example, Math
// Vector*, and Matrix* classes compile fine, but not Integrate1 or
// BSplineRectangle.  So the following macro is *not* defined for this
// platform.  If you have a Linux system that does appear to require the
// instantiation before, then enable this macro.
// #define WML_INSTANTIATE_BEFORE

// common standard library headers
#include <cassert>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <sys/stat.h>

#endif
//----------------------------------------------------------------------------

namespace Wml
{

class WML_ITEM System
{
public:
    // little/big endian support
    static void SwapBytes (int iSize, void* pvValue);
    static void SwapBytes (int iSize, int iQuantity, void* pvValue);
    static void EndianCopy (int iSize, const void* pvSrc, void* pvDst);
    static void EndianCopy (int iSize, int iQuantity, const void* pvSrc,
        void* pvDst);

    static unsigned int MakeRGB (unsigned char ucR, unsigned char ucG,
        unsigned char ucB);

    static unsigned int MakeRGBA (unsigned char ucR, unsigned char ucG,
        unsigned char ucB, unsigned char ucA);

    // time utilities
    static double GetTime ();

    // TO DO.  Pathname handling to access files in subdirectories.
    static bool FileExists (const char* acFilename);

    // convenient utilities
    static bool IsPowerOfTwo (int iValue);
};

// allocation and deallocation of 2D arrays
template <class T> void Allocate2D (int iCols, int iRows, T**& raatArray);
template <class T> void Deallocate2D (T** aatArray);

// allocation and deallocation of 3D arrays
template <class T> void Allocate3D (int iCols, int iRows, int iSlices,
    T***& raaatArray);
template <class T> void Deallocate3D (int iRows, int iSlices,
    T*** aaatArray);

#include "WmlSystem.inl"
#include "WmlSystem.mcr"

}

#endif

