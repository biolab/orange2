// Magic Software, Inc.
// http://www.magic-software.com
// http://www.wild-magic.com
// Copyright (c) 2004.  All Rights Reserved
//
// The Wild Magic Library (WML) source code is supplied under the terms of
// the license agreement http://www.magic-software.com/License/WildMagic.pdf
// and may not be copied or disclosed except in accordance with the terms of
// that agreement.

#include "WmlSystem.h"
using namespace Wml;

//----------------------------------------------------------------------------
void System::SwapBytes (int iSize, void* pvValue)
{
    assert( iSize >= 1 );
    if ( iSize == 1 )
        return;

    // size must be even
    assert( (iSize & 1) == 0 );

    char* acBytes = (char*) pvValue;
    for (int i0 = 0, i1 = iSize-1; i0 < iSize/2; i0++, i1--)
    {
        char cSave = acBytes[i0];
        acBytes[i0] = acBytes[i1];
        acBytes[i1] = cSave;
    }
}
//----------------------------------------------------------------------------
void System::SwapBytes (int iSize, int iQuantity, void* pvValue)
{
    assert( iSize >= 1 );
    if ( iSize == 1 )
        return;

    // size must be even
    assert( (iSize & 1) == 0 );

    char* acBytes = (char*) pvValue;
    for (int i = 0; i < iQuantity; i++, acBytes += iSize)
    {
        for (int i0 = 0, i1 = iSize-1; i0 < iSize/2; i0++, i1--)
        {
            char cSave = acBytes[i0];
            acBytes[i0] = acBytes[i1];
            acBytes[i1] = cSave;
        }
    }
}
//----------------------------------------------------------------------------
bool System::IsPowerOfTwo (int iValue)
{
    return (iValue != 0) && ((iValue & -iValue) == iValue);
}
//----------------------------------------------------------------------------
bool System::FileExists (const char* acFilename)
{
    FILE* pkFile = fopen(acFilename,"r");
    if ( pkFile )
    {
        fclose(pkFile);
        return true;
    }
    return false;
}
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// Microsoft Windows
//----------------------------------------------------------------------------
#if defined(WIN32)
#include <windows.h>
//----------------------------------------------------------------------------
void System::EndianCopy (int iSize, const void* pvSrc, void* pvDst)
{
    memcpy(pvDst,pvSrc,iSize);
}
//----------------------------------------------------------------------------
void System::EndianCopy (int iSize, int iQuantity, const void* pvSrc,
    void* pvDst)
{
    memcpy(pvDst,pvSrc,iSize*iQuantity);
}
//----------------------------------------------------------------------------
unsigned int System::MakeRGB (unsigned char ucR, unsigned char ucG,
    unsigned char ucB)
{
    return (ucR | (ucG << 8) | (ucB << 16) | (0xFF << 24));
}
//----------------------------------------------------------------------------
unsigned int System::MakeRGBA (unsigned char ucR, unsigned char ucG,
    unsigned char ucB, unsigned char ucA)
{
    return (ucR | (ucG << 8) | (ucB << 16) | (ucA << 24));
}
//----------------------------------------------------------------------------
double System::GetTime ()
{
    // 64-bit quantities
    LARGE_INTEGER iFrequency, iCounter;

    QueryPerformanceFrequency(&iFrequency);
    QueryPerformanceCounter(&iCounter);
    return ((double)iCounter.QuadPart)/((double)iFrequency.QuadPart);
}
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// Macintosh OS X
//----------------------------------------------------------------------------
#elif defined(__APPLE__)
//----------------------------------------------------------------------------
void System::EndianCopy (int iSize, const void* pvSrc, void* pvDst)
{
    memcpy(pvDst,pvSrc,iSize);
    SwapBytes(iSize,pvDst);
}
//----------------------------------------------------------------------------
void System::EndianCopy (int iSize, int iQuantity, const void* pvSrc,
    void* pvDst)
{
    memcpy(pvDst,pvSrc,iSize*iQuantity);
    SwapBytes(iSize,iQuantity,pvDst);
}
//----------------------------------------------------------------------------
unsigned int System::MakeRGB (unsigned char ucR, unsigned char ucG,
    unsigned char ucB)
{
    return (0xFF | (ucB << 8) | (ucG << 16) | (ucR << 24));
}
//----------------------------------------------------------------------------
unsigned int System::MakeRGBA (unsigned char ucR, unsigned char ucG,
    unsigned char ucB, unsigned char ucA)
{
    return (ucA | (ucB << 8) | (ucG << 16) | (ucR << 24));
}
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// SGI IRIX
//----------------------------------------------------------------------------
#elif defined(SGI_IRIX)
//----------------------------------------------------------------------------
void System::EndianCopy (int iSize, const void* pvSrc, void* pvDst)
{
    memcpy(pvDst,pvSrc,iSize);
}
//----------------------------------------------------------------------------
void System::EndianCopy (int iSize, int iQuantity, const void* pvSrc,
    void* pvDst)
{
    memcpy(pvDst,pvSrc,iSize*iQuantity);
}
//----------------------------------------------------------------------------
unsigned int System::MakeRGB (unsigned char ucR, unsigned char ucG,
    unsigned char ucB)
{
    return (ucR | (ucG << 8) | (ucB << 16) | (0xFF << 24));
}
//----------------------------------------------------------------------------
unsigned int System::MakeRGBA (unsigned char ucR, unsigned char ucG,
    unsigned char ucB, unsigned char ucA)
{
    return (ucR | (ucG << 8) | (ucB << 16) | (ucA << 24));
}
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// HP-UX
//----------------------------------------------------------------------------
#elif defined(HP_UX)
//----------------------------------------------------------------------------
void System::EndianCopy (int iSize, const void* pvSrc, void* pvDst)
{
    memcpy(pvDst,pvSrc,iSize);
}
//----------------------------------------------------------------------------
void System::EndianCopy (int iSize, int iQuantity, const void* pvSrc,
    void* pvDst)
{
    memcpy(pvDst,pvSrc,iSize*iQuantity);
}
//----------------------------------------------------------------------------
unsigned int System::MakeRGB (unsigned char ucR, unsigned char ucG,
    unsigned char ucB)
{
    return (ucR | (ucG << 8) | (ucB << 16) | (0xFF << 24));
}
//----------------------------------------------------------------------------
unsigned int System::MakeRGBA (unsigned char ucR, unsigned char ucG,
    unsigned char ucB, unsigned char ucA)
{
    return (ucR | (ucG << 8) | (ucB << 16) | (ucA << 24));
}
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// Sun SOLARIS
//----------------------------------------------------------------------------
#elif defined(SUN_SOLARIS)
//----------------------------------------------------------------------------
void System::EndianCopy (int iSize, const void* pvSrc, void* pvDst)
{
    memcpy(pvDst,pvSrc,iSize);
}
//----------------------------------------------------------------------------
void System::EndianCopy (int iSize, int iQuantity, const void* pvSrc,
    void* pvDst)
{
    memcpy(pvDst,pvSrc,iSize*iQuantity);
}
//----------------------------------------------------------------------------
unsigned int System::MakeRGB (unsigned char ucR, unsigned char ucG,
    unsigned char ucB)
{
    return (ucR | (ucG << 8) | (ucB << 16) | (0xFF << 24));
}
//----------------------------------------------------------------------------
unsigned int System::MakeRGBA (unsigned char ucR, unsigned char ucG,
    unsigned char ucB, unsigned char ucA)
{
    return (ucR | (ucG << 8) | (ucB << 16) | (ucA << 24));
}
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// Linux
//----------------------------------------------------------------------------
#else
//----------------------------------------------------------------------------
void System::EndianCopy (int iSize, const void* pvSrc, void* pvDst)
{
    memcpy(pvDst,pvSrc,iSize);
}
//----------------------------------------------------------------------------
void System::EndianCopy (int iSize, int iQuantity, const void* pvSrc,
    void* pvDst)
{
    memcpy(pvDst,pvSrc,iSize*iQuantity);
}
//----------------------------------------------------------------------------
unsigned int System::MakeRGB (unsigned char ucR, unsigned char ucG,
    unsigned char ucB)
{
    return (ucR | (ucG << 8) | (ucB << 16) | (0xFF << 24));
}
//----------------------------------------------------------------------------
unsigned int System::MakeRGBA (unsigned char ucR, unsigned char ucG,
    unsigned char ucB, unsigned char ucA)
{
    return (ucR | (ucG << 8) | (ucB << 16) | (ucA << 24));
}
//----------------------------------------------------------------------------
#endif
//----------------------------------------------------------------------------
