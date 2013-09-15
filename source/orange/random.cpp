#include "values.hpp" // must include it because we include root.hpp...

#include "random.ppp"

TRandomGenerator *_globalRandom;
PRandomGenerator globalRandom;

void random_cpp_gcUnsafeInitialization() 
{ _globalRandom = mlnew TRandomGenerator();
  globalRandom = PRandomGenerator(_globalRandom);
}
