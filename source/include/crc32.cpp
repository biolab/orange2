unsigned long crc_table[256];

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
{ 
  for(int i = 0; i <= 0xFF; i++) {
    crc_table[i] = reflect(i, 8) << 24; 
    for (int j = 0; j < 8; j++) 
      crc_table[i] = (crc_table[i] << 1) ^ (crc_table[i] & (1 << 31) ? 0x04c11db7 : 0); 
    crc_table[i] = reflect(crc_table[i], 32); 
  }

  return true;
} 

bool __f = initCRC32();
