###
# 1. exports the orange version tagged with TAG
# 2. compiles orange sources
# 3. builds rpm
#
# TODO:
#    - list of files to be excluded from RPM
###

TAG=stable
SPWD=`pwd`

echo exporting version using tag: $TAG

rm -Rf orange Genomics
if ! cvs -d :pserver:cvso@estelle.fri.uni-lj.si:/CVS -Q export -r $TAG -f orange; then
  echo ERROR exporting orange
  exit 1
fi

##cvs -d :pserver:cvso@estelle.fri.uni-lj.si:/CVS -Q export -r $TAG -f source
if ! cvs -d :pserver:cvso@estelle.fri.uni-lj.si:/CVS -Q export -r $TAG -f Genomics; then
  echo ERROR exporting Genomics
  exit 2
fi

echo -e "\ncompiling Orange:"
cd source
if make; then 
  echo OK
  cd $SPWD
  mv orange.so orange
  mv statc.so orange
  mv corn.so orange
else
  echo -e "\n\nERROR compiling source"
  echo returning into directory $SPWD
  cd $SPWD
  exit 1
fi

## make RPM
## stable

## snapshot

