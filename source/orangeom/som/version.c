#include <stdio.h>
#include "version.h"
static char version[200];

char *get_version()
{
  sprintf(version, "%s R%d compiled: %s %s", SOM_VERSION, SOM_REVISION, 
	  __DATE__, __TIME__);

  return version;
}
