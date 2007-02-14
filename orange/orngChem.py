import orange
import sys
from openeye.oechem import *
from openeye.oedepict import *
import sets

class Set(sets.Set):
    pass

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
    OEAssignAromaticFlags(mol)
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
    return rev

def stripFirstAtom(smiles):
    if len(smiles)>2:
        return smiles[2:], smiles[:2]
    else:
        return smiles, ""
    
def stripLastAtom(smiles):
    if len(smiles)>2:
        return smiles[:-2], smiles[-2:]
    else:
        return smiles, ""

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

def extendFragments(fragments,sets):
    extended=Set()
    for i in range(len(fragments)):
        for j in range(i,len(fragments)):
            first=fragments[i]
            _second=fragments[j]
            if first in sets and _second in sets:
                newset=sets[first].intersection(sets[_second])
            else:
                newset=None
            second, end=stripLastAtom(_second)
            ind=first.find(second)
            if ind>1:
                new=first+end
                sets[new]=newset
                sets[reverseSmiles(new)]=newset
                extended.add(new)
            
            second, end=stripLastAtom(reverseSmiles(fragments[j]))
            ind=first.find(second)
            if ind>1:
                new=first+end
                sets[new]=newset
                sets[reverseSmiles(new)]=newset
                extended.add(new)

            second, start=stripFirstAtom(fragments[j])
            ind=first.find(second)
            if ind==0:
                new=start+first
                sets[new]=newset
                sets[reverseSmiles(new)]=newset
                extended.add(new)
            
            second, start=stripFirstAtom(reverseSmiles(fragments[j]))
            ind=first.find(second)
            if ind==0:
                new=start+first
                sets[new]=newset
                sets[reverseSmiles(new)]=newset
                extended.add(start+first)
    #print [decodeSmiles(s) for s in extended]
    return extended

def extendFragmentsSimple(fragments, atomSymbols):
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

def filterByFrequency(fragments, freq, data, sets):
    new=[]
    unmatched=[]
    pat=OESubSearch()
    for f in fragments:
        ss=sets.get(f, None) or sets.get(reverseSmiles(f), None) or range(len(data))
        set=[]
        pat.Init(decodeSmiles(f))
        count=0
        for i in ss:
            if pat.SingleMatch(data[i]):
                count+=1
                set.append(i)
        s=Set(set)
        sets[f]=s
        sets[reverseSmiles(f)]=s
        if float(count)/len(data)>freq:
            new.append(f)
        else:
            unmatched.append(f)
    return new, unmatched

def filterByOccurence(fragments, data, sets):
    return filterByFrequency(fragments, 0, data, sets)

def updateSpecific(G,S,c, cache={}):
    setsDict={}
    first, f, data=c
    candidateSymbols=filterOutUnusedSymbols(AtomSymbols, data)
    data=[molGraph(d) for d in data]
    filterFunc=lambda g: freq(g, data)>f
    #G=filter(filterFunc, G)
    G=filterByFrequency(G,f,data,setsDict)
    C=Set(candidateSymbols)
    F=[]
    F.append([""])
    F.append(filter(filterFunc, C))
    while(C):
        #print "Loop updateSpecific"
        #print "extFrag"
        if len(F)>2:
            C=extendFragments(F[-1], setsDict)
        else:
            C=extendFragmentsSimple(F[-1], candidateSymbols)
        #print "filter general"
        C=filter(lambda c: moreGeneral(c, S), C)
        #print "filter dup"
        C=removeDuplicates(C)
        print "filter freq",len(C)
        F.append(filterByFrequency(C,f,data,setsDict))
        #print C
        #print F
    UF=[]
    for f in F:
        UF.extend(f)
    UF=removeDuplicates(UF)
    print "filtering S", len(UF)
    UF=filterMaxSpecific(UF)
    S=filter(lambda s: moreSpecific(s, G), UF)
    #S=filter(lambda s: (not moreGeneral(s, UF)) and moreSpecific(s, G), UF)
    print "S:",[decodeSmiles(s) for s in S]
    return G,S

def updateGeneral(G,S,c):
    setsDict={}
    first, f, data=c
    candidateSymbols=filterOutUnusedSymbols(AtomSymbols, data)
    data=[molGraph(d) for d in data]
    filterFunc=lambda s: freq(s, data)>f
    S=filter(lambda s:freq(s,data)<=f, S)
    C=Set(candidateSymbols)
    F=[]
    I=[]
    F.append([""])
    F.append(filterByFrequency(C,f,data,setsDict))
    I.append(C.difference(Set(F[-1])))
    while C:
        if len(F)>2:
            C=extendFragments(F[-1], setsDict)
        else:
            C=extendFragmentsSimple(F[-1], candidateSymbols)
        #print "filter general"
        C=Set(filter(lambda c: moreGeneral(c, S), C))
        #print "filter dup"
        C=Set(removeDuplicates(C))
        print "filter freq",len(C)
        F.append(filterByFrequency(C,f,data,setsDict))
        I.append(C.difference(Set(F[-1])))
        #print C
        #print F
    UI=[]
    for i in I:
        UI.extend(i)
    UI=removeDuplicates(UI)
    print "filtering G", len(UI)
    UI=filterMaxGeneral(UI)
    G=filter(lambda g: moreGeneral(g,S), UI)
    #G=filter(lambda g: (not moreSpecific(g, UI)) and moreGeneral(g,S), UIr)
    print "G:",[decodeSmiles(g) for g in G]
    return G,S

def extractFragments(genBorder, specBorder, ignoreGeneral=False):
    if ignoreGeneral:
        frag=[]
        for s in specBorder:
            for i in range(0, len(s),2):
                for j in range(1,len(s)-i+2,2):
                    frag.append(s[i:i+j])
        frag=removeDuplicates(frag)
        frag.sort()
        frag.reverse()
        return frag
    else:
        frag=[]
        for s in specBorder:
            for i in range(0,len(s),2):
                for j in range(1,len(s)-i+2,2):
                    st=s[i:i+j]
                    rst=reverseString(st)
                    for g in genBorder:
                        if g in st or g in rst:
                            frag.append(st)
                            break
        return frag

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

def p_chisq(value):
    lookup=[(0.0000393, 0.005),(0.000157,0.01),(0.00982, 0.25),(0.0158,0.1),(0.0642,0.2),(0.45, 0.5),(1.642,0.8),(2.71,0.9),(3.84,0.95),(5.02,0.975),(6.65,0.99),(7.88,0.995)]
    cur=0.0
    for v,p in lookup:
        if v<value:
            cur=p
    return cur

def __lazar_learn__(trainingSet, testStruct=""):
    setsDict={}
    testSetsDict={}
    candidateSymbols=filterOutUnusedSymbols(AtomSymbols, [t[0] for t in trainingSet]+[testStruct])
    data=[molGraph(d[0]) for d in trainingSet]
    testData=[molGraph(testStruct)]
    G=[]
    unknown=[]
    C=Set(candidateSymbols)
    F=[]
    s1,s2=filterByOccurence(C, testData, testSetsDict)
    s1,s2=filterByOccurence(s1,data, setsDict)
    unknown.extend(s2)
    F.append(s1)
    while(C):
        if len(F)>1:
            C=extendFragments(F[-1], setsDict)
        else:
            C=extendFragmentsSimple(F[-1], candidateSymbols)
        C=removeDuplicates(C)
        s1,s2=filterByOccurence(C,testData,testSetsDict)
        s1,s2=filterByOccurence(s1,data, setsDict)
        unknown.extend(s2)
        F.append(s1)
    UF=[]
    for f in F:
        UF.extend(f)
    UF=removeDuplicates(UF)
    nActive=float(len(filter(lambda c:c[1], trainingSet)))
    nAll=float(len(trainingSet))
    for f in UF:
        s=setsDict[f]
        s.fAll=float(len(s))
        s.fActive=float(len(filter(lambda c:trainingSet[c][1], list(s))))
        s.fInactive=float(len(s)-s.fActive)
        s.pActive=s.fActive/s.fAll-nActive/nAll
        s.pChisq=p_chisq(nAll*(abs(s.fActive*(nAll-nActive)-(s.fAll-s.fActive)*nActive)-nAll/2)**2/(nAll*s.fAll*(nAll+s.fAll)*(nAll-nActive+s.fAll-s.fActive)))
        s.pFinal=s.pActive*s.pChisq
    #print all fragments
    print "\npredicting: ",testStruct
    """print "all fragments"
    for f in UF:
        s=setsDict[f]
        print decodeSmiles(f),  s.fActive, s.fInactive, s.pActive, s.pChisq, s.pFinal"""
    #filter redundant fragments
    l=[]
    for f in UF:
        s=setsDict[f]
        for ff in UF:
            redundant=False
            ss=setsDict[ff]
            if f!=ff and (s.issubset(ss) or s.issuperset(ss)) and (abs(s.pFinal)<abs(ss.pFinal) or (s.pFinal==ss.pFinal and f<ff)):
                redundant=True
                break
        if not redundant:
            l.append(f)
    print "non-redundant fragments"
    for f in l:
        s=setsDict[f]
        print decodeSmiles(f),  s.fActive, s.fInactive, s.pActive, s.pChisq, s.pFinal
    print "Unknown fragments: ", [decodeSmiles(d) for d in unknown]
    s=sum([setsDict[f].pFinal for f in l])
    sa=sum([setsDict[f].pFinal for f in l if setsDict[f].pFinal>0])
    si=sum([setsDict[f].pFinal for f in l if setsDict[f].pFinal<0])
    sall=sum([abs(setsDict[f].pFinal) for f in l ]) or 1e-6
    print "prediction: ",s, "(",sa/sall,",",abs(si/sall),")"
    #print setsDict[encodeSmiles("N-c:c:c:c:c-C=O")], setsDict[encodeSmiles("N-c:c:c:c:c-C")]
    return s>0 and 1 or 0

def testAcc(data):
    c=0
    num=len(data)
    for d in data[:num]:
        train=list(data)
        train.remove(d)
        if __lazar_learn__(train, d[0])==d[1]:
            c+=1
    print float(c)/num
        
########################################################
##Visualization
########################################################

def moleculeFragment2BMP(molSmiles, fragSmiles, filename, size=200, title=""):
    """given smiles codes of molecle and a fragment will draw the molecule and save it
    to a file"""
    mol=OEGraphMol()
    OEParseSmiles(mol, molSmiles)
    depict(mol)
    mol.SetTitle(title)
    match=subsetSearch(mol, fragSmiles)
    view=createMolView(mol, size)
    colorSubset(view, mol, match)
    renderImage(view, filename)

def molecule2BMP(molSmiles, filename, size=200, title=""):
    """given smiles code of a molecule will draw the molecule and save it
    to a file"""
    mol=OEGraphMol()
    OEParseSmiles(mol, molSmiles)
    mol.SetTitle(title)
    depict(mol)
    view=createMolView(mol, size)
    renderImage(view, filename)

def depict(mol):
    """depict a molecule - i.e assign 2D coordinates to atoms"""
    if mol.GetDimension()==3:
        OEPerceiveChiral(mol)
        OE3DToBondStereo(mol)
        OE3DToAtomStereo(mol)
    OEAddDepictionHydrogens(mol)
    OEDepictCoordinates(mol)
    OEMDLPerceiveBondStereo(mol)
    OEAssignAromaticFlags(mol)

def subsetSearch(mol, pattern):
    """finds the matches of pattern in mol"""
    pat=OESubSearch()
    pat.Init(pattern)
    return pat.Match(mol,1)

def createMolView(mol, size=200, title=""):
    """creates a view for the molecule mol"""
    view=OEDepictView()
    view.SetMolecule(mol)
    view.SetLogo(False)
    view.SetTitleSize(12)
    view.AdjustView(size, size)
    return view

def colorSubset(view, mol, match):
    """assigns a differnet color to atoms and bonds of mol in view that are present in match"""
    for matchbase in match:
        for mpair in matchbase.GetAtoms():
            style=view.AStyle(mpair.target.GetIdx())
            #set style
            style.r=255
            style.g=0
            style.b=0

    for matchbasem in match:
        for mpair in matchbase.GetBonds():
            style=view.BStyle(mpair.target.GetIdx())            
            #set style
            style.r=255
            style.g=0
            style.b=0

def renderImage(view, filename):
    """renders the view to a filename"""
    img=OE8BitImage(view.XRange(), view.YRange())
    view.RenderImage(img)
    ofs=oeofstream(filename)
    OEWriteBMP(ofs, img)

def render2OE8BitImage(view):
    """renders the view to a OE8BitImage"""
    img=OE8BitImage(view.XRange(), view.YRange())
    view.RenderImage(img)
    return view
        
def testLazar():
    import orange
    d=orange.ExampleTable("E:\chem\mutagen_raw.tab")
    data=[(str(e["SMILES"]), int(e[-1])) for e in d]
    #print data
    testAcc(data)
    """__lazar_learn__(data[1:],data[0][0])
    __lazar_learn__(data[2:],data[1][0])
    __lazar_learn__(data[3:],data[2][0])
    __lazar_learn__(data[4:],data[3][0])"""
    #__lazar_learn__(data[5:],"Oc1ccc2ccccc2c1N=Nc3ccccc3")
    #__lazar_learn__(data[5:],"Cc1ccc2C(=O)c3ccccc3C(=O)c2c1N(=O)O")

def test():
    import orange
    d=orange.ExampleTable("E:\chem\mutagen_raw.tab")
    active=[str(e["SMILES"]) for e in d if e[-1]==1]
    inactive=[str(e["SMILES"]) for e in d if e[-1]==0]
    #frag=find_fragments(active, 0.2, inactive, 0.2)
    frag=find_fragments(active, 0.05, inactive, 0.05)
    #frag=find_fragments(inactive, 0.1, active, 0.1)
    #for f in frag:
    #    print f, freq(encodeSmiles(f), [molGraph(g) for g in active]),freq(encodeSmiles(f), [molGraph(g) for g in inactive])
    file=open("fragments.txt", "w")
    file.writelines([f+"\n" for f in frag])
    file.close()
    
if __name__=="__main__":
    testLazar()
