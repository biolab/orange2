c:
cd \Python25\Lib\site-packages

rem svn co --force http://www.ailab.si/svn/orange/trunk/orange/ orange
rem svn co --force http://www.ailab.si/svn/orange/trunk/add-ons/Bioinformatics/ orange/add-ons/Bioinformatics
rem svn co --force http://www.ailab.si/svn/orange/trunk/add-ons/Text/ orange/add-ons/Text
rem svn co --force http://www.ailab.si/svn/orange/trunk/testing/regressionTests/ regressionTests

svn cleanup orange
svn update --force orange

svn cleanup orange/add-ons/Bioinformatics
svn update --force orange/add-ons/Bioinformatics

svn cleanup orange/add-ons/Text
svn update --force orange/add-ons/Text

svn cleanup regressionTests
svn update --force regressionTests

python orange/downloadPyd.py

cd regressionTests

python xtest.py test --module=orange > regression_tests_orange_log.txt
python xtest.py report --module=orange > regression_tests_orange_report.txt
python xtest.py report-html --module=orange > regression_tests_orange_report.html

python xtest.py test --module=obi > regression_tests_obi_log.txt
python xtest.py report --module=obi > regression_tests_obi_report.txt
python xtest.py report-html --module=obi > regression_tests_obi_report.html

python xtest.py test --module=text > regression_tests_text_log.txt
python xtest.py report --module=text > regression_tests_text_report.txt
python xtest.py report-html --module=text > regression_tests_text_report.html

rm -rf Z:\Volumes\download\regressionLogs\winxp\*

cp -rf results Z:\Volumes\download\regressionLogs\winxp
cp -f regression*.txt Z:\Volumes\download\regressionLogs\winxp\
cp -f regression*.html Z:\Volumes\download\regressionLogs\winxp\

shutdown -s
