#ifdef _DARWIN_BUNDLE

extern "C" void initorangeLow();

extern "C" void initorange() {
  initorangeLow();
}

#endif