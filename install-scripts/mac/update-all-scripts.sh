#!/bin/bash
#
# Should be run as: ./update-all-scripts.sh
#

if [ ! -e /private/tmp/repos ]; then
	mkdir /private/tmp/repos
fi

LOCAL_REPO=/private/tmp/repos/orange

if [ ! -e $LOCAL_REPO ]; then
	hg clone https://bitbucket.org/biolab/orange $LOCAL_REPO
fi

hg pull --update -R $LOCAL_REPO

cp ${LOCAL_REPO}/install-scripts/mac/*.sh ./

chmod +x *.sh

# Zero exit value
true
