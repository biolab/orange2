import orange
import sys
from openeye.oechem import *
from sets import Set

class MostSpecific:
    def __contains__(self, anything):
        return True
    def __len__(self):
        return sys.maxint

AtomSymbols=["[He]","[Li]","[Be]","B","C","c","N","n","O","o","F","[Ne]","[Na]","[Mg]","[Al]","[Si]","P","p","S","s","Cl","[Ar]",
             "[K]","[Ca]","[Sc]","[Ti]","[V]","[Cr]","[Mn]","[Fe]","[Co]","[Ni]","[Cu]","[Zn]","[Ga]","[Ge]","[As]","[Se]","Br",
             "[Kr]","[Rb]","[Sr]","[Y]","[Zr]","[Nb]","[Mo]","[Tc]","[Ru]","[Rh]","[Pd]","[Ag]","[Cd]","[In]","[Sn]","[Sb]","[Te]",
             "I","[Xe]","[Cs]","[Ba]","[Hf]","[Ta]","[W]","[Re]","[Os]","[Ir]","[Pt]","[Au]","[Hg]","[Tl]","[Pb]","[Bi]","[Po]",
             "[At]","[Rn]","[Fr]","[Ra]","[Rf]","[Db]","[Sg]","[Bh]","[Hs]","[Mt]","[Ds]","[Rg]","[Uub]","[Uut]","[Uuq]","[Uup]",
             "[Uuh]","[Uus]","[Uuo]","[La]","[Ce]","[Pr]","[Nd]","[Pm]","[Sm]","[Eu]","[Tb]","[Dy]","[Ho]","[Er]","[Tm]","[Yb]",
             "[Lu]","[Ac]","[Th]","[Pa]","[U]","[Np]","[Pu]","[Am]","[Cm]","[Bk]","[Cf]","[Es]","[Fm]","[Md]","[No]","[Lr]"]

AtomicBondSymbols=["-","=",":","#"]

AtomicSymbolCodeDict=dict([(text, chr(i+1)) for i,text in enumerate(AtomSymbols+AtomicBondSymbols)])
ReverseAtomicSymbolCodeDict=dict([(val, key) for key, val in AtomicSymbolCodeDict.items()])

def molGraph(smiles):
    mol=OEGraphMol()
    OEParseSmiles(mol, smiles)
    return mol

def encodeSmiles(smiles):
    code=""
    while smiles:
        if smiles.startswith("["):
            i=smiles.index("]")
            code+=AtomicSymbolCodeDict[smiles[:i+1]]
            smiles=smiles[i+1:]
        elif smiles.startswith("Br"):
            code+=AtomicSymbolCodeDict["Br"]
            smiles=smiles[2:]
        elif smiles.startswith("Cl"):
            code+=AtomicSymbolCodeDict["Cl"]
            smiles=smiles[2:]
        else:
            code+=AtomicSymbolCodeDict[smiles[0]]
            smiles=smiles[1:]
    return code

def decodeSmiles(code):
    smiles=""
    for c in code:
        smiles+=ReverseAtomicSymbolCodeDict[c]
    return smiles

def reverseString(string):
    rev=""
    for c in string:
        rev=c+rev
    return rev

def reverseSmiles(smiles):
    rev=""
    #print smiles
    while smiles:
        if smiles.startswith("["):
            i=smiles.index("]")
            rev=smiles[:i+1]+rev
            smiles=smiles[i+1:]
        elif smiles.startswith("Br"):
            rev="Br"+rev
            smiles=smiles[2:]
        elif smiles.startswith("Cl"): 
            rev="Cl"+rev
            smiles=smiles[2:]
        else:
            rev=smiles[0]+rev
            smiles=smiles[1:]
    #print rev
    return rev

pat=OESubSearch()

def freq(f, data):
    if f=="": return 1
    elif f.__class__==MostSpecific: return 0
    pat.Init(decodeSmiles(f))
    count=0
    for d in data:
        if pat.SingleMatch(d):
            count+=1
    return float(count)/len(data)

def filterOutUnusedSymbols(symbols, data):
    tt="".join(data)
    return [encodeSmiles(s) for s in filter(lambda s:s in tt, symbols)]
    tt="".join([encodeSmiles(d) for d in data])
    sym=[encodeSlims(s) for s in symbols]
    sym=filter(lambda s: s in tt, sym)
    return [decodeSlims(s) for s in sym]

def extendFragments(fragments, atomSymbols):
    extended=Set()
    for f in fragments:
        for bond in AtomicBondSymbols:
            for atom in atomSymbols:
                extended.add("%s%s%s"%(f,AtomicSymbolCodeDict[bond],atom))
                extended.add("%s%s%s"%(atom,AtomicSymbolCodeDict[bond],f))
    return extended

def moreGeneral(gen, spec):
    revGen=reverseString(gen)
    for s in spec:
        #if len(gen)>len(s): continue
        if gen in s: return True
        elif revGen in s: return True
    return False

def moreSpecific(spec, gen):
    revSpec=reverseString(spec)
    for g in gen:
        #if len(g)<len(spec): continue
        if g in spec: return True
        elif g in revSpec: return True
    return False

def removeDuplicates(smiles):
    r=[]
    for i in range(len(smiles)):
        a=smiles.pop()
        if a not in smiles and reverseString(a) not in smiles:
            r.append(a)
    return r

def filterMaxSpecific(fragments):
    fragments=list(fragments)
    fragments.sort(lambda a,b: -cmp(len(a), len(b)))
    #print fragments
    ret=[]
    while fragments:
        a=fragments.pop(-1)
        if not moreGeneral(a, fragments):
            ret.append(a)
    return ret

def filterMaxGeneral(fragments):
    fragments=list(fragments)
    fragments.sort(lambda a,b:cmp(len(a), len(b)))
    #print fragments
    ret=[]
    while fragments:
        a=fragments.pop(-1)
        if not moreSpecific(a, fragments):
            ret.append(a)
    return ret

def updateSpecific(G,S,c, cache={}):
    first, f, data=c
    candidateSymbols=filterOutUnusedSymbols(AtomSymbols, data)
    data=[molGraph(d) for d in data]
    filterFunc=lambda g: freq(g, data)>f
    G=filter(filterFunc, G)
    C=Set(candidateSymbols)
    F=[]
    F.append([""])
    F.append(filter(filterFunc, C))
    while(C):
        #print "Loop updateSpecific"
        #print "extFrag"
        C=extendFragments(F[-1], candidateSymbols)
        #print "filter general"
        C=filter(lambda c: moreGeneral(c, S), C)
        #print "filter dup"
        C=removeDuplicates(C)
        print "filter freq",len(C)
        F.append(filter(filterFunc,C))
        #print C
        #print F
    UF=Set()
    for f in F:
        UF.union_update(f)
    print "filtering S", len(UF)
    UF=filterMaxSpecific(UF)
    S=filter(lambda s: moreSpecific(s, G), UF)
    #S=filter(lambda s: (not moreGeneral(s, UF)) and moreSpecific(s, G), UF)
    print "S:",[decodeSmiles(s) for s in S]
    return G,S

def updateGeneral(G,S,c):
    first, f, data=c
    candidateSymbols=filterOutUnusedSymbols(AtomSymbols, data)
    data=[molGraph(d) for d in data]
    filterFunc=lambda s: freq(s, data)>f
    S=filter(lambda s:freq(s,data)<=f, S)
    C=Set(candidateSymbols)
    F=[]
    I=[]
    F.append([""])
    F.append(filter(filterFunc, C))
    I.append(C.difference(Set(F[-1])))
    while C:
        C=extendFragments(F[-1], candidateSymbols)
        #print "filter general"
        C=Set(filter(lambda c: moreGeneral(c, S), C))
        #print "filter dup"
        C=Set(removeDuplicates(C))
        print "filter freq",len(C)
        F.append(filter(filterFunc, C))
        I.append(C.difference(Set(F[-1])))
        #print C
        #print F
    UI=Set()
    for i in I:
        UI.union_update(i)
    print "filtering G", len(UI)
    UI=filterMaxGeneral(UI)
    G=filter(lambda g: moreGeneral(g,S), UI)
    #G=filter(lambda g: (not moreSpecific(g, UI)) and moreGeneral(g,S), UIr)
    print "G:",[decodeSmiles(g) for g in G]
    return G,S

def extractFragments(genBorder, specBorder):
    frag=[]
    for s in specBorder:
        for i in range(0, len(s),2):
            for j in range(1,len(s)-i+2,2):
                frag.append(s[i:i+j])
    frag=removeDuplicates(frag)
    frag.sort()
    frag.reverse()
    return frag
    """
    for s in specBorder:
        for i in range(0,len(s),2):
            for j in range(1,len(s)-i+2,2):
                st=s[i:i+j]
                rst=reverseString(st)
                for g in genBorder:
                    if g in st or g in rst:
                        frag.append(st)
                        break
    return frag"""

def fragment_search(query=[]):
    """query is a list of constraints in a form ("<"|">", f, data) meaning freq(fragment, data)<|>f  i.e. that frequency of
    a fragment in data (a list of SMILES) must be lower|higher"""
    GeneralBorder=[""]
    SpecificBorder=[MostSpecific()]
    for q in query:
        if q[0]=="<":
            GeneralBorder, SpecificBorder=updateGeneral(GeneralBorder, SpecificBorder, q)
        elif q[0]==">":
            GeneralBorder, SpecificBorder=updateSpecific(GeneralBorder, SpecificBorder, q)
    frag=removeDuplicates(extractFragments(GeneralBorder, SpecificBorder))
    frag=[decodeSmiles(f) for f in frag]
    
    GeneralBorder=[decodeSmiles(g) for g in GeneralBorder]
    SpecificBorder=[decodeSmiles(s) for s in SpecificBorder]
    print "G:", GeneralBorder
    print "S:", SpecificBorder
    print "Fragments:", frag
    return frag

def find_fragments(active, activeFreq, inactive=[], inactiveFreq=0):
    """active and inactive are lists of smiles.
    activeFreq and inactiveFreq are floats from [0..1].
    Finds the fragments that ocur with a freqency higher the activeFreq in active compounds
    and lower then inactiveFreq in inactive compounds"""
    if inactive and inactiveFreq:
        return fragment_search([(">", activeFreq, active), ("<", inactiveFreq, inactive)])
    else:
        return fragment_search([(">", activeFreq, active)])
        

def map_fragments(fragments, smiles, binary=True):
    """Returns a dictionary with smiles codes as keys. The items are also dictionaries
    with fragment codes as keys, items are [0,1] if binary is True else, number of ocurances of
    this fragments in the coresponding chemical"""
    ret={}
    pat=OESubSearch()
    for s in smiles:
        c=molGraph(s)
        d={}
        for f in fragments:
            pat.Init(f)
            count=0
            for m in pat.Match(c,1):
                count+=1
            if binary:
                d[f]=count!=0 and 1 or 0
            else:
                d[f]=count
        ret[s]=d
    return ret

def test():
    import orange
    d=orange.ExampleTable("mutagen_raw.tab")
    active=[str(e["SMILES"]) for e in d if e[-1]==1]
    inactive=[str(e["SMILES"]) for e in d if e[-1]==0]
    #frag=find_fragments(active, 0.5, inactive, 0.5)
    frag=find_fragments(active, 0.01, inactive, 0.01)
    #frag=find_fragments(inactive, 0.1, active, 0.1)
    for f in frag:
        print f, freq(encodeSmiles(f), [molGraph(g) for g in active]),freq(encodeSmiles(f), [molGraph(g) for g in inactive])
    file=open("fragments.txt", "w")
    file.writelines([f+"\n" for f in frag])
    file.close()
    
if __name__=="__main__":
    test()
