/* This code is adapted from http://www.createwindow.com/programming/crc32/,
   Copyright 2000 - 2003 Richard A. Ellingson. */

unsigned long crc32_table[256];

long reflect(unsigned long ref, char ch) 
{ unsigned long value(0); 

  for(int i = 1; i < (ch + 1); i++) {
    if(ref & 1) 
      value |= 1 << (ch - i); 
    ref >>= 1; 
  } 
  return value; 
} 


bool initCRC32() 
{ unsigned long ulPolynomial = 0x04c11db7; 

  for(int i = 0; i <= 0xFF; i++) {
    crc32_table[i] = reflect(i, 8) << 24; 
    for (int j = 0; j < 8; j++) 
      crc32_table[i] = (crc32_table[i] << 1) ^ (crc32_table[i] & (1 << 31) ? ulPolynomial : 0); 
    crc32_table[i] = reflect(crc32_table[i], 32); 
  }

  return true;
} 

bool __f = initCRC32();
