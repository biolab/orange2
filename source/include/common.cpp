/*
    This file is part of Orange.

    Orange is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    Orange is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Orange; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Authors: Janez Demsar, Blaz Zupan, 1996--2002
    Contact: janez.demsar@fri.uni-lj.si
*/


#ifdef _MSC_VER
 #include <limits>
 using namespace std;
 #ifdef _STLP_LIMITS
  #define _STLP_FLOAT_INF_REP { 0, 0x7f80 }
  #define _STLP_FLOAT_QNAN_REP { 0, 0xffc0 }
  #define _STLP_FLOAT_SNAN_REP { 0x5555, 0x7f85 }

  #define _STLP_DOUBLE_INF_REP { 0, 0, 0, 0x7ff0 }
  #define _STLP_DOUBLE_QNAN_REP { 0, 0, 0, 0xfff8 }
  #define _STLP_DOUBLE_SNAN_REP { 0x5555, 0x5555, 0x5555, 0x7ff5 }

  const _F_rep _LimG<bool>::_F_inf = _STLP_FLOAT_INF_REP;
  const _F_rep _LimG<bool>::_F_qNaN = _STLP_FLOAT_QNAN_REP;
  const _F_rep _LimG<bool>::_F_sNaN = _STLP_FLOAT_SNAN_REP;

  //const _D_rep _LimG<bool>::_D_inf = _STLP_DOUBLE_INF_REP;
  const _D_rep _LimG<bool>::_D_qNaN = _STLP_DOUBLE_QNAN_REP;
  const _D_rep _LimG<bool>::_D_sNaN = _STLP_DOUBLE_SNAN_REP;
 #endif
#endif
