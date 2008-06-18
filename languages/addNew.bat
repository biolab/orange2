c:\python23\Tools\i18n\pygettext.py -d new %1 %2 %3 %4 %5 %6 %7 %8 %9
msgcat messages.pot new.pot -o messages.pot
msgmerge -U slovenian.po messages.pot
