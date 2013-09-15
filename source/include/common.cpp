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
