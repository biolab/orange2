/* Nothing special here.

   We just need to include initialization.px somewhere,
   and define the two functions it calls.
*/


#include "mymodule_globals.hpp"

bool initmymoduleExceptions()
{ return true; }

void gcmymoduleUnsafeStaticInitialization()
{}

#include "px/initialization.px"