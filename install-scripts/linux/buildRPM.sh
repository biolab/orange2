TAG=stable
VER=$TAG

## check out from CVS
## and create the distribution (Source) file with the top-level directory of same name
ORANGESOURCE=orange-linux-$VER.tgz
ORANGEDIR=orange-$VER

#rm -Rf $ORANGEDIR 
#cvs -d :pserver:cvso@estelle.fri.uni-lj.si:/CVS -Q export -r $TAG -f -d orange-$VER orange
#cvs -d :pserver:cvso@estelle.fri.uni-lj.si:/CVS -Q export -r $TAG -f -d orange-$VER/source source
#tar -cvzf $ORANGESOURCE orange-$VER
#rm -Rf $ORANGEDIR

## create .spec file for building RPM for Orange version: $VER
SPECF=orange.spec

## build spec file, then use it to create RPM
# preamble section
echo Summary: Orange is Data Mining Application with Visual Programming capabilities > $SPECF
echo Name: orange >> $SPECF
echo Version: $TAG >> $SPECF
echo Release: 1 >> $SPECF
echo Copyright: GPL >> $SPECF
echo Group: Applications >> $SPECF
echo Source: http://www.ailab.si/Orange/Download/$ORANGESOURCE>> $SPECF
echo URL: http://www.ailab.si/Orange >> $SPECF
##echo Distribution: >> $SPECF
echo Icon: orange.gif >> $SPECF
echo Vendor: AI lab, University of Ljubljana, Slovenia >> $SPECF
echo Packager: orange@george.fri.uni-lj.si >> $SPECF
echo "Requires: qt >= 2.3, python = 2.3" >> $SPECF
echo BuildRoot: /tmp/orange >> $SPECF 
echo >> $SPECF
echo "%description" >> $SPECF
echo "Orange is a component based machine learning library for Python\n \\
developed at Laboratory of artificial inteligence, Faculty of\n \\
Computer and Information Science, University of Ljubljana, Slovenia." >> $SPECF
echo  >> $SPECF

# prep section
echo "%prep" >> $SPECF
echo cd /usr/src/redhat/BUILD/orange-stable >> $SPECF
#echo "%setup" >> $SPECF

# build section
echo "%build" >> $SPECF
echo cd /usr/src/redhat/BUILD/orange-stable/source >> $SPECF
echo python makedep.py >> $SPECF
echo make >> $SPECF

# install section
echo "%install" >> $SPECF
echo cd /usr/src/redhat/BUILD/orange-stable/source >> $SPECF
echo make ROOT="\$RPM_BUILD_ROOT" install >> $SPECF

# files section
echo %files >> $SPECF
echo /usr/local/lib/python2.3/site-packages/orange >> $SPECF
echo /usr/local/bin/canvas >> $SPECF
echo %docdir /usr/local/doc/orange >> $SPECF
echo /usr/local/doc/orange >> $SPECF
echo %config /usr/local/lib/python2.3/site-packages/orange.pth >> $SPECF
#echo %config /usr/local/lib/python2.3/site-packages/orange/OrangeCanvas/widgetregistry.xml >> $SPECF


# post script
# %post
# call ldconfig

## spec file built, now use it to make RPM
# copy spec file into appropriate Build Directory
cp $SPECF /usr/src/redhat/SPECS
cp orange.gif /usr/src/redhat/SOURCES

# copy sources into the appropriate Build Directory
cp $ORANGESOURCE /usr/src/redhat/SOURCES

# call rpm
rpmbuild -ba $SPECF ## stored into /var/tmp   ## --buildarch i486
#
