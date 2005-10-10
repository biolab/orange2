cd /home/orange
cvs update
chmod +x ./install-scripts/linux/downloadinstallregress.sh ./install-scripts/linux/testOrange.sh

cd /home/orange/daily
rm -Rf orange 
cvs -d :pserver:cvso@estelle.fri.uni-lj.si:/cvs checkout -d orange orange
cvs -d :pserver:cvso@estelle.fri.uni-lj.si:/cvs checkout -d orange/source source
cp /home/orange/install-scripts/linux/setup.py /home/orange/daily/orange

cd /home/orange/daily/orange
echo `date` > ../output.log
if ! python setup.py compile >> ../output.log 2>&1 ; then
  cat compiling.log >> ../output.log
  mail -s "Linux: ERROR compiling Orange" tomaz.curk@fri.uni-lj.si < ../output.log
  cat ../output.log
  echo -e "\n\nERROR compiling, see log above"
  exit 1
fi


