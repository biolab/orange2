/home/orange/mount_estelleDownload

TMP=`grep SOURCE_SNAPSHOT /mnt/estelleDownload/filenames.set`
export ${TMP}

rm -rf /home/orange/test_download
mkdir /home/orange/test_download
cd /home/orange/test_download

tar zxpvf /mnt/estelleDownload/${SOURCE_SNAPSHOT}

cd orange

if ! python setup.py compile &> output.log; then
  mail -s "Linux: ERROR compiling Orange" janez.demsar@fri.uni-lj.si < output.log
  mail -s "Linux: ERROR compiling Orange" tomaz.curk@fri.uni-lj.si < output.log
  mail -s "Linux: ERROR compiling Orange" jurem@flextronics.si < output.log 
  cat output.log
  echo -e "\n\nERROR compiling, see log above"
  
  /home/orange/umount_estelleDownload
  exit 1 
fi

python setup.py install --orangepath=/home/orange/test_download/install &> install.log

echo -e "\n\n"
echo -e "\n\nStarting regression tests...\n\n" >> output.log

if ! ~/testOrange.sh >> output.log 2>&1; then
  cd /home/orange/test_download/orange
  cat output.log >> install.log
  mv install.log output.log
  mail -s "Linux: ERROR regression tests (compile OK) Orange" janez.demsar@fri.uni-lj.si < output.log
  mail -s "Linux: ERROR regression tests (compile OK) Orange" tomaz.curk@fri.uni-lj.si < output.log
  mail -s "Linux: ERROR regression tests (compile ok) Orange" jurem@flextronics.si < output.log 
  cat output.log
  echo regression tests failed
else
  cd /home/orange/test_download/orange
  cat output.log >> install.log
  mv install.log output.log
  mail -s "Linux: Orange compiled successfully" tomaz.curk@fri.uni-lj.si < output.log
  mail -s "Linux: Orange compiled successfully" jurem@flextronics.si < output.log 
  cat output.log
  echo compiling was successful
fi

/home/orange/umount_estelleDownload
