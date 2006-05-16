/* Nothing special here.

   We just need to include initialization.px somewhere,
   and define the two functions it calls.
*/


#include "mymodule_globals.hpp"

bool initcoreExceptions()
{ return true; }

void gccoreUnsafeStaticInitialization()
{}

#include "px/initialization.px"