# Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
# Description:
#	icons
import sys, string
import orange, os.path

from qt import *
iDocIndex = 1 

canvasPicsDir = os.path.join(os.path.split(os.path.abspath(orange.__file__))[0], "OrangeCanvas/icons")

file_new  = os.path.join(canvasPicsDir, "doc.png")
output    = os.path.join(canvasPicsDir, "output.png")
file_open = os.path.join(canvasPicsDir, "open.png")
file_save = os.path.join(canvasPicsDir, "save.png")
file_print= os.path.join(canvasPicsDir, "print.png")
file_exit = os.path.join(canvasPicsDir, "exit.png")
move_left = os.path.join(canvasPicsDir, "moveleft.png")
move_right= os.path.join(canvasPicsDir, "moveright.png")

"""
file_open = [
	'16 13 5 1',
	'. c #040404',
	'# c #808304',
	'a c None',
	'b c #f3f704',
	'c c #f3f7f3',
	'aaaaaaaaa...aaaa',
	'aaaaaaaa.aaa.a.a',
	'aaaaaaaaaaaaa..a',
	'a...aaaaaaaa...a',
	'.bcb.......aaaaa',
	'.cbcbcbcbc.aaaaa',
	'.bcbcbcbcb.aaaaa',
	'.cbcb...........',
	'.bcb.#########.a',
	'.cb.#########.aa',
	'.b.#########.aaa',
	'..#########.aaaa',
	'...........aaaaa'
]

file_save = [
	'14 14 4 1',
	'. c #040404',
	'# c #808304',
	'a c #bfc2bf',
	'b c None',
	'..............',
	'.#.aaaaaaaa.a.',
	'.#.aaaaaaaa...',
	'.#.aaaaaaaa.#.',
	'.#.aaaaaaaa.#.',
	'.#.aaaaaaaa.#.',
	'.#.aaaaaaaa.#.',
	'.##........##.',
	'.############.',
	'.##.........#.',
	'.##......aa.#.',
	'.##......aa.#.',
	'.##......aa.#.',
	'b.............'
]

file_print = [
	'16 14 6 1',
	'. c #000000',
	'# c #848284',
	'a c #c6c3c6',
	'b c #ffff00',
	'c c #ffffff',
	'd c None',
	'ddddd.........dd',
	'dddd.cccccccc.dd',
	'dddd.c.....c.ddd',
	'ddd.cccccccc.ddd',
	'ddd.c.....c....d',
	'dd.cccccccc.a.a.',
	'd..........a.a..',
	'.aaaaaaaaaa.a.a.',
	'.............aa.',
	'.aaaaaa###aa.a.d',
	'.aaaaaabbbaa...d',
	'.............a.d',
	'd.aaaaaaaaa.a.dd',
	'dd...........ddd'
]


file_new = [
"16 16 3 1",
" 	c None",
".	c #000000000000",
"X	c #FFFFFFFFFFFF",
"				",
"   ......	   ",
"   .XXX.X.	  ",
"   .XXX.XX.	 ",
"   .XXX.XXX.	",
"   .XXX.....	",
"   .XXXXXXX.	",
"   .XXXXXXX.	",
"   .XXXXXXX.	",
"   .XXXXXXX.	",
"   .XXXXXXX.	",
"   .XXXXXXX.	",
"   .XXXXXXX.	",
"   .........	",
"				",
"				"
]
"""