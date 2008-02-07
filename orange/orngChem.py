"""A library for searching frequent molecular fragments (substructures) based on the
Mining Molecular Fragments: Finding Relevant Substructures of Molecules
Christian Borgelt and Michael R. Berthold.
Classes (see their corresponding __doc__ strings for further detail):
    Fragment        : Representation of the fragment
    FragmentMiner   : The main class that does the search
    Fragmenter      : A class that is used to fragment an ExampleTable
    FragmentBasedLearner    : A learner wrapper class that first runs the molecular fragmentation on the data
"""
from openbabel import OBMol, OBAtom, OBBond, OBSmartsPattern, OBConversion, OBMolAtomIter, OBMolBondIter, OBAtomBondIter
##from pybel import *
from copy import deepcopy
import orange


debug=False
try:
    from pywin.debugger import set_trace
except:
    def setTrace(): pass

class Atom(object):
    def __init__(self, molecule, atomicNum=6, aromaticFlag=False, atomIndex=0):
        self.molecule=molecule
        self.atomicNum=atomicNum
        self.aromaticFlag=aromaticFlag
        self.atomIndex=atomIndex
        self.extendedIndex=0
        self.lastExtendedBondOrder=1
        self.lastExtendedAtomicNum=1
        self.molecule.AddAtom(self)
    def GetAtomicNum(self): return self.atomicNum
    def IsAromatic(self): return self.aromaticFlag
    def Match(self, atom):
        #Match this atom to an OBAtom
        if self.atomicNum==atom.GetAtomicNum() and self.IsAromatic()==atom.IsAromatic():
            return True
        else:
            return False
    def GetIdx(self):
        return self.atomIndex

class Bond(object):
    def __init__(self, molecule, atom1=None, atom2=None, bondOrder=1):
        self.molecule=molecule
        self.atom1=atom1
        self.atom2=atom2
        self.bondOrder=bondOrder
        self.molecule.AddBond(self)
    def GetBondOrder(self): return self.bondOrder
    def GetNbrAtom(self, atom):
        if atom==self.atom1:
            return self.atom2
        elif atom==self. atom2:
            return  self.atom1
        else:
            raise Exception("Atom does not belong to this bond")
    def Match(self, bond):
        #Match this bond to an OBBond
        if self.GetbondOrder()==bond.GetBondOrder():
            return True
        else:
            return False

class Ring(object):
    def __init__(self, molecule, obRing, embeding):
        self.molecule=molecule
        molecule.AddRing(self)
        self.ringAtoms=[]
        self.isAromatic=obRing.IsAromatic()
        reverseEmbedingDict=embeding.GetReverseDict()
        newAtoms={}
        self.extendedIndex=oldNextExtendedIndex=molecule.nextExtendedIndex
        for ind1, ind2 in GetRingPairs(obRing._path):
            atom1=embeding.GetReverseEmbededAtom(ind1, None) or newAtoms.get(ind1, None)
            if not atom1:
                atom1=Atom(molecule, embeding.molecule.GetAtom(ind1).GetAtomicNum(), obRing.IsAromatic())
                newAtoms[ind1]=atom1
                embeding[atom1.atomIndex]=ind1
            atom2=embeding.GetReverseEmbededAtom(ind2, None) or newAtoms.get(ind2, None)
            if not atom2:
                atom2=Atom(molecule, embeding.molecule.GetAtom(ind2).GetAtomicNum(), obRing.IsAromatic())
                newAtoms[ind2]=atom2
                embeding[atom2.atomIndex]=ind2
            Bond(molecule, atom1, atom2, embeding.molecule.GetBond(ind1, ind2).GetBondOrder())
            self.ringAtoms.append(atom1)
        for atom in newAtoms.values():
            atom.extendedIndex=oldNextExtendedIndex
        self.extendedIndex=oldNextExtendedIndex
        molecule.nextExtendedIndex=oldNextExtendedIndex+1
    def Size(self):
        return len(self.ringAtoms)
    def __getattr__(self, name):
        if name == "_path":
            return [a.GetIdx() for a in self.ringAtoms]
        raise AttributeError 

class Molecule(object):
    def __init__(self):
        self.atoms=[]
        self.bonds=[]
        self.rings=[]
        self.nextExtendedIndex=0

    def __deepcopy__(self, memo):
        mol=Molecule()
        memo[id(self)]=mol
        mol.__in_place_deepcopy__(self, memo)
        return mol
    def __in_place_deepcopy__(self, mol, memo):
        if debug: print "Copying molecule"
        self.atoms=deepcopy(mol.atoms, memo)
        #memo[mol.atoms]=self.atoms
        self.bonds=deepcopy(mol.bonds, memo)
        #memo[mol.bonds]=self.bonds
        self.rings=deepcopy(mol.rings, memo)
        self.nextExtendedIndex=mol.nextExtendedIndex
        #memo[mol.rings]=self.rings"""
        
    def AddAtom(self, atom):
        if atom in self.atoms:
            raise Exception("Atom already present")
        self.atoms.append(atom)
        atom.atomIndex=len(self.atoms)-1
        atom.extendedIndex=self.nextExtendedIndex
        self.nextExtendedIndex+=1
    def AddBond(self, bond):
        if bond in self.bonds:
            raise Exception("Bond already present")
        self.bonds.append(bond)
        bond.bondIndex=len(self.bonds)-1
    def AddRing(self, ring):
        if ring in self.rings:
            raise Exception("Ring already present")
        self.rings.append(ring)
        ring.ringIndex=len(self.rings)-1
    def GetBond(self, atom1, atom2):
        for bond in self.bonds:
            if bond.atom1==atom1 and bond.atom2==atom2:
                return bond
            if bond.atom1==atom2 and bond.atom2==atom1:
                return bond
        return None
    def GetAtom(self, index):
        return self.atoms[index]
    
def cmpAtoms(atom1, atom2):
    return atom1.GetAtomicNum()==atom2.GetAtomicNum() and atom1.IsAromatic()==atom2.IsAromatic()
def cmpAtomBonds(bond1, bond2):
    return bond1.IsAromatic() and bond2.IsAromatic() or bond1.GetBondOrder()==bond2.GetBondOrder()

class Embeding(dict):
    def __init__(self, embeding={}, molecule=None, fragment=None):
        dict.__init__(self, embeding)
        self.molecule=molecule
        self.fragment=fragment
        if embeding.__class__==Embeding:
            if not self.molecule:
                self.molecule=embeding.molecule
            if not self.fragment:
                self.fragment=embeding.fragment
    def __deepcopy__(self, memo):
        if id(self.fragment) in memo:
            return Embeding(self, fragment=memo[id(self.fragment)])
        else:
            return Embeding(self)
    def GetEmbededAtom(self, atomInd):
        return self.molecule.GetAtom(self.__getitem__(atomInd))
    def GetReverseEmbededAtom(self, atomInd, default=None):
        rev=self.GetReverseDict()
        if atomInd in rev:
            return self.fragment.atoms[rev[atomInd]]
        else:
            return default
    def GetReverseDict(self):
        return dict([(value, key) for key, value in self.items()])

def GetRingPairs(list):
    return reduce(lambda a,b:a+[(a[-1][1],b)], list[1:], [(list[-1],list[0])])

class FragmentExtension(object):
    def __init__(self, startAtomInd, embedings):
        self.startAtomInd=startAtomInd
        self.embedings=embedings
    def IsEquivalent(self, extension):
        pass
    def MergeFrom(self, extension):
        pass
    def Extend(self, fragment):
        pass

class FragmentExtensionByAtom(FragmentExtension):
    def __init__(self, atomicNum, aromatic, atomIndices, embedings):
        self.atomicNum=atomicNum
        self.aromatic=aromatic
        self.atomIndices=atomIndices
        self.embedings=embedings
    def IsEquivalent(self, extension):
        return self.atomicNum==extension.atomicNum and self.aromatic==extension.aromatic
    def MergeFrom(self, extension):
        self.embedings.extend(extension.embedings)
        self.atomIndices.extend(extension.atomIndices)
    def Extend(self, fragment):
        atom=Atom(fragment, self.atomicNum, self.aromatic)
        embedings=[Embeding(e, fragment=fragment)for e in self.embedings]
        for embeding, atomInd in zip(embedings, self.atomIndices):
            embeding[atom.atomIndex]=atomInd
        fragment.embedings=embedings
        
class FragmentExtensionByBondAtom(FragmentExtension):
    def __init__(self, startAtomInd, bondOrder, endAtoms, embedings):
        FragmentExtension.__init__(self, startAtomInd, embedings)
        self.bondOrder=bondOrder
        self.endAtoms=endAtoms
    def IsEquivalent(self, extension):
        return self.startAtomInd==extension.startAtomInd and self.bondOrder==extension.bondOrder and cmpAtoms(self.endAtoms[0], extension.endAtoms[0])
    def MergeFrom(self, extension):
        self.endAtoms.extend(extension.endAtoms)
        self.embedings.extend(extension.embedings)
    def Extend(self, fragment):
        if debug: print "Extending by bond and atom"
        sAtom=fragment.atoms[self.startAtomInd]
        atom=Atom(fragment, self.endAtoms[0].GetAtomicNum(),self.endAtoms[0].IsAromatic())
        #atom.SetAtomicNum(self.endAtoms[0].GetAtomicNum())
        
        Bond(fragment, sAtom, atom, self.bondOrder)
        sAtom.lastExtendedBondOrder=self.bondOrder
        sAtom.lastExtendedAtomicNum=atom.GetAtomicNum()
        fragment.lastExtendedAtomIndex=sAtom.extendedIndex
        embedings=[Embeding(e, fragment=fragment)for e in self.embedings]
        for embeding, endAtom in zip(embedings, self.endAtoms):
            embeding[atom.atomIndex]=endAtom.GetIdx()
        fragment.embedings=embedings
        
class FragmentExtensionByBond(FragmentExtension):
    def __init__(self, startAtomInd, bondOrder, endAtomInd, embedings):
        FragmentExtension.__init__(self, startAtomInd, embedings)
        self.bondOrder=bondOrder
        self.endAtomInd=endAtomInd
    def IsEquivalent(self, extension):
        return self.startAtomInd==extension.startAtomInd and self.bondOrder==extension.bondOrder and self.endAtomInd==extension.endAtomInd
    def MergeFrom(self, extension):
        self.embedings.extend(extension.embedings)
    def Extend(self, fragment):
        if debug: print "Extending by bond"
        sAtom=fragment.atoms[self.startAtomInd]
        eAtom=fragment.atoms[self.endAtomInd]
        Bond(fragment, sAtom, eAtom, self.bondOrder)
        sAtom.lastExtendedBondOrder=self.bondOrder
        sAtom.lastExtendedAtomicNum=eAtom.GetAtomicNum()
        fragment.lastExtendedAtomIndex=sAtom.extendedIndex
        fragment.embedings=[Embeding(e, fragment=fragment) for e in self.embedings]

class FragmentExtensionByBondRing(FragmentExtension):
    def __init__(self, startAtomInd, bondOrder, endAtoms, rings, embedings):
        FragmentExtension.__init__(self, startAtomInd, embedings)
        self.bondOrder=bondOrder
        self.endAtoms=endAtoms
        self.rings=rings
    def GetRingMapping(self, endAtom1, ring1, embeding1, endAtom2, ring2, embeding2):
        rDict1=embeding1.GetReverseDict()
        rDict2=embeding2.GetReverseDict()
        ringMappings=[]
        i=list(ring1._path).index(endAtom1.GetIdx())
        c0=list(ring1._path)[i:]+list(ring1._path)[:i]
        i=list(ring2._path).index(endAtom2.GetIdx())
        c1=list(ring2._path)[i:]+list(ring2._path)[:i]
        l=list(c1[1:]) #l=list(c1[:-1])
        l.reverse()
        c2=[c1[0]]+l #c2=[c1[-1]]+l
        for c in [c1,c2]:
            mapping=[]
            for (ra11, ra12), (ra21, ra22) in zip(GetRingPairs(map(lambda i:ring1.GetParent().GetAtom(i), c0)),GetRingPairs(map(lambda i:ring2.GetParent().GetAtom(i), c))):
                if cmpAtoms(ra12, ra22) and cmpAtomBonds(ring1.GetParent().GetBond(ra11, ra12), ring2.GetParent().GetBond(ra21, ra22)):
                    mapping.append((ra12.GetIdx(), ra22.GetIdx()))
                else:
                    break
            else:
                ringMappings.append(dict(mapping))
        return ringMappings
    def IsEquivalent(self, extension):
        if self.startAtomInd==extension.startAtomInd and self.bondOrder==extension.bondOrder and len(self.rings[0]._path)==len(extension.rings[0]._path) \
           and self.rings[0].fingerprint==extension.rings[0].fingerprint:
            return bool(self.GetRingMapping(self.endAtoms[0], self.rings[0], self.embedings[0], extension.endAtoms[0], extension.rings[0], extension.embedings[0]))
        return False
    def MergeFrom(self, extension):
        self.endAtoms.extend(extension.endAtoms)
        self.embedings.extend(extension.embedings)
        self.rings.extend(extension.rings)
    def Extend(self, fragment):
        if debug: print "Extending by ring and bond"
        sAtom=fragment.atoms[self.startAtomInd]
        endAtom1=self.endAtoms[0]
        ring1=self.rings[0]    #One ring to rule them all
        embeding1=Embeding(self.embedings[0], fragment=fragment)
        #Add all the atoms and bonds in the ring
        newRing=Ring(fragment, ring1, embeding1) #Changes embeding
        Bond(fragment, fragment.atoms[self.startAtomInd], fragment.atoms[embeding1.GetReverseDict()[endAtom1.GetIdx()]], self.bondOrder)
        sAtom.lastExtendedAtomicNum=endAtom1.GetAtomicNum()
        sAtom.lastExtendedBondOrder=self.bondOrder
        fragment.lastExtendedAtomIndex=sAtom.extendedIndex
        #Update embedings
        #embeding1=self.embedings[0]
        fragment.embedings=[]
        for endAtom, ring, embeding in zip(self.endAtoms, self.rings, self.embedings):
            #embeding[sAtom.atomIndex]=endAtom.GetIdx()
            mappings=self.GetRingMapping(endAtom1, ring1, embeding1, endAtom, ring, embeding)
            for map in mappings:
                embeding=Embeding(embeding, fragment=fragment)
                for atom in newRing.ringAtoms:
                    embeding[atom.atomIndex]=map[embeding1[atom.atomIndex]]
                embeding[newRing]=ring
                fragment.embedings.append(embeding)
        
class FragmentExtensionByRing(FragmentExtension):
    def __init__(self, rings, embedings):
        FragmentExtension.__init__(self, 0, embedings)
        self.rings=rings
    def GetRingMapping(self, ring1, embeding1, ring2, embeding2):
        rDict1=embeding1.GetReverseDict()
        rDict2=embeding2.GetReverseDict()
        ringMappings=[]
        for i in range(ring2.Size()):
            c1=list(ring2._path)[i:]+list(ring2._path)[:i]
            l=list(c1[1:])
            l.reverse()
            c2=[c1[0]]+l
            for c in [c1,c2]:
                mapping=[]
                for (ra11, ra12), (ra21, ra22) in zip(GetRingPairs(map(lambda i:ring1.GetParent().GetAtom(i), ring1._path)),GetRingPairs(map(lambda i:ring2.GetParent().GetAtom(i), c))):
                    if cmpAtoms(ra12, ra22) and cmpAtomBonds(ring1.GetParent().GetBond(ra11, ra12), ring2.GetParent().GetBond(ra21, ra22)) and rDict1.get(ra12.GetIdx(), None)==rDict2.get(ra22.GetIdx(), None):
                        mapping.append((ra12.GetIdx(), ra22.GetIdx()))
                    else:
                        break
                else:
                    ringMappings.append(dict(mapping))
        return ringMappings
    def IsEquivalent(self, extension):
        if len(self.rings[0]._path)==len(extension.rings[0]._path) and self.rings[0].fingerprint==extension.rings[0].fingerprint:
            return bool(self.GetRingMapping(self.rings[0], self.embedings[0], extension.rings[0], extension.embedings[0]))
        return False
    def MergeFrom(self, extension):
        self.rings.extend(extension.rings)
        self.embedings.extend(extension.embedings)
    def Extend(self, fragment):
        if debug: print "Extending by ring"
        ring1=self.rings[0]
        tmpEmbeding=Embeding(self.embedings[0],fragment=fragment)
        newRing=Ring(fragment, ring1, tmpEmbeding) #Changes embeding
        embeding1=self.embedings[0]
##        fragment.lastExtendedAtomIndex=sAtom.extendedIndex
        fragment.embedings=[]
        for ring, embeding in zip(self.rings, self.embedings):
            mappings=self.GetRingMapping(ring1, embeding1, ring, embeding)
            for map in mappings:
                embeding=Embeding(embeding, fragment=fragment)
                for atom in newRing.ringAtoms:
                    embeding[atom.atomIndex]=map[tmpEmbeding[atom.atomIndex]]
                embeding[newRing]=ring
                fragment.embedings.append(embeding)
        
class Fragment(Molecule):
    """A class representing a molecular fragment
    Methods:
        ToOBMol()   : Returns an openbabel.OBMol object representation
        ToSmiles()  : Returns a SMILES code representation
        ToCanonicalSmiles() : Returns a canonical SMILES code representation
        Support()   : Returns the support of the fragment in the active set
        OcurrencesIn(smiles): Returns the number of times a fragment is containd
                    in the molecule represented by the smiles code argument
        ContainedIn(smiles) : Returns True if the fragment is present in the molecule
                    represented by the smiles code argument
    """
    writer=OBConversion()
    writer.SetInAndOutFormats("smi","smi")
    def __init__(self, miner=None, excludeAtomList=[]):
        Molecule.__init__(self)
        self.embedings=[]
        self.miner=miner
        self.excludeAtomList=excludeAtomList
        self.lastExtendedAtomicNum=0
        self.lastExtendedAtomIndex=0

    def __deepcopy__(self, memo):
        f=Fragment()
        memo[id(self)]=f
        Molecule.__in_place_deepcopy__(f, self, memo)
        f.embedings=[Embeding(e,fragment=f) for e in self.embedings]
        f.miner=self.miner
        f.excludeAtomList=self.excludeAtomList
        f.lastExtendedAtomIndex=self.lastExtendedAtomIndex
        return f
        
    def InitializeFragment(self, atomicNum):
        mol=OBMol()
        atom=Atom(self, atomicNum)
        for mol in self.miner.GetAllMolecules():
            for a in OBMolAtomIter(mol):
                if atom.Match(a):
                    self.embedings.append(Embeding({atom.atomIndex : a.GetIdx()}, molecule=mol, fragment=self))
                    
    def IsAtomExcluded(self, atom):
        return atom.GetAtomicNum() in self.excludeAtomList
    
    def GetCandidateAtoms(self):
        return filter(lambda a:a.extendedIndex>=self.lastExtendedAtomIndex, self.atoms)

    def GetCandidateRings(self):
        return filter(lambda r:r.extendedIndex>=self.lastExtendedAtomIndex, self.rings)

    def FilterRings(self, candidateRings, embeding):
        for ring1 in candidateRings:
            for ring2 in filter(lambda r:type(r)==type(ring1), embeding.values()):
                if ring1.this==ring2.this:
                    break
            else:
                yield ring1
        
    def GetCandidatesFromAtom(self, atomInd, embeding):
        candidates=[]
        reverseEmbedingDict=embeding.GetReverseDict()
        atom=embeding.GetEmbededAtom(atomInd)
        if atom.GetParent().NumHvyAtoms()==1:
            return candidates
        for bond in OBAtomBondIter(atom):   #crashes if the atom is alone in a molecule
            nbrAtom=bond.GetNbrAtom(atom)
            if self.IsAtomExcluded(nbrAtom):
                    continue
            if bond.GetBondOrder()>self.atoms[atomInd].lastExtendedBondOrder or (bond.GetBondOrder()==self.atoms[atomInd].lastExtendedBondOrder and nbrAtom.GetAtomicNum()>=self.atoms[atomInd].lastExtendedAtomicNum):
                if self.miner.addWholeRings: #Whole rings are added at the same time (no new bond can connect to an atom already in the embeding)
                    if nbrAtom.IsInRing():
                        if not bond.IsInRing(): #ring to ring extensions are handled in GetRingCandidatesFromRing 
                            for ring in self.FilterRings(self.miner.rings[embeding.molecule], embeding):
                                if ring.IsMember(nbrAtom):
                                    candidates.append(FragmentExtensionByBondRing(atomInd, bond.GetBondOrder(), [nbrAtom], [ring], [embeding]))
##                        elif len(self.atoms)==1: #
##                            for ring in self.FilterRings(self.miner.rings[embeding.molecule], embeding):
##                                if ring.IsMember(nbrAtom) and ring.IsMember(atom) and ring not in addedRings:
##                                    candidates.append(FragmentExtensionByRing([ring], [embeding]))
##                                    addedRings.append(ring)
                           
                    elif nbrAtom.GetIdx() not in reverseEmbedingDict:
                        candidates.append(FragmentExtensionByBondAtom(atomInd, bond.GetBondOrder(), [nbrAtom], [embeding]))                                
                else: 
                    if nbrAtom.GetIdx() in reverseEmbedingDict:
                        if not self.GetBond(self.GetAtom(atomInd), self.GetAtom(reverseEmbedingDict[nbrAtom.GetIdx()])) :
                            candidates.append(FragmentExtensionByBond(atomInd, bond.GetBondOrder(), reverseEmbedingDict[nbrAtom.GetIdx()], [embeding]))
                    else:
                        candidates.append(FragmentExtensionByBondAtom(atomInd, bond.GetBondOrder(), [nbrAtom], [embeding]))
        #candidates.sort(lambda a,b:cmp(a[0],b[0]) or cmp(a[1].GetAtomicNum(),b[1].GetAtomicNum()))
        return candidates

    def GetRingCandidatesFromRing(self, ring, embeding):
        candidates=[]
        reverseEmbedingDict=embeding.GetReverseDict()
        for candidateRing in self.FilterRings(self.miner.rings[embeding.molecule], embeding):
            for atom in ring.ringAtoms:
                if candidateRing.IsMember(embeding.GetEmbededAtom(atom.atomIndex)):
                    candidates.append(FragmentExtensionByRing([candidateRing], [embeding]))
                    break
        return candidates                             
    
    def GetCandidatesFromEmbeding(self, embeding):
        candidates=[]
        if len(self.atoms)==1 and self.miner.addWholeRings:
            atom=embeding.GetEmbededAtom(self.atoms[0].GetIdx())
            if atom.IsInRing():
                for ring in embeding.molecule.rings:
                    if ring.IsMember(atom):
                        candidates.append(FragmentExtensionByRing([ring],[embeding]))
                return candidates
            
        if debug: print "Parsing candidate bonds"
        for atom in self.GetCandidateAtoms():
            candidates.extend(self.GetCandidatesFromAtom(atom.atomIndex, embeding))
        
        if debug: print "Parsing candidate rings"
        for ring in self.GetCandidateRings():
            candidates.extend(self.GetRingCandidatesFromRing(ring, embeding))            
        return candidates        
        
    def Extend(self):
        if debug :print "Extending"
        candidates=[]
        for embeding in self.embedings:
            c=self.GetCandidatesFromEmbeding(embeding)
            candidates.extend(c)
            #embedingDict[embeding]=c
            
        #group equivalent candidates
        if debug: print "Grouping candidates"
        groups=[]
        for extension in candidates:
            for ext in groups:
                if ext.__class__==extension.__class__ and ext.IsEquivalent(extension):
                    ext.MergeFrom(extension)
                    break
            else:
                groups.append(extension)
  
        #generate new fragments
        if debug: print "Generating new fragments"
        newFragments=[]
        for extension in groups:
            #set_trace()
            f=deepcopy(self)
            extension.Extend(f)
            newFragments.append(f)
        return newFragments
    
    def ToOBMol(self):
        atomCache={}
        mol=OBMol()
        mol.BeginModify()
        for sourceAtom in self.atoms:
            atom=mol.NewAtom()
            atom.SetAtomicNum(sourceAtom.GetAtomicNum())
            if sourceAtom.IsAromatic():
                atom.SetAromatic()
##                atom.SetSpinMultiplicity(2)
            atomCache[sourceAtom]=atom
        for sourceBond in self.bonds:
            mol.AddBond(atomCache[sourceBond.atom1].GetIdx(), atomCache[sourceBond.atom2].GetIdx(), sourceBond.GetBondOrder())
##        mol.SetAromaticPerceived()
        mol.AssignSpinMultiplicity()
##        mol.UnsetAromaticPerceived()
        mol.EndModify()
        return mol

    def ToSmiles(self):
        writer=OBConversion()
        writer.SetInAndOutFormats("smi", "smi")
        return writer.WriteString(self.ToOBMol()).strip()
    
    def ToCannonicalSmiles(self):
        atomCache={}
        mol=OBMol()
        for sourceAtom in self.atoms:
            atom=mol.NewAtom()
            atom.SetAtomicNum(sourceAtom.GetAtomicNum())
            if sourceAtom.IsAromatic():
                atom.SetAromatic()
                atom.SetSpinMultiplicity(2)
            atomCache[sourceAtom]=atom
        for sourceBond in self.bonds:
            mol.AddBond(atomCache[sourceBond.atom1].GetIdx(), atomCache[sourceBond.atom2].GetIdx(), sourceBond.GetBondOrder())
        writer=OBConversion()
        writer.SetInAndOutFormats("smi", "can")
        return writer.WriteString(mol).strip()

    def Support(self, activeSet=None):
        activeSet=self.miner.activeSet if activeSet==None else activeSet
        uniqueMolecules=set()
        for embeding in self.embedings:
            if embeding.molecule in activeSet:
                uniqueMolecules.add(embeding.molecule)
##        s=set(filter(lambda mol:self.ContainedIn(mol), self.miner.GetAllMolecules()))
##        if len(s) != len(uniqueMolecules):
##            writer=OBConversion()
##            writer.SetInAndOutFormats("smi", "smi")
##            print "\n",self.ToSmiles()
##            for m in uniqueMolecules: print writer.WriteString(m).strip()
##            for m in s: print writer.WriteString(m).strip()
        
        return float(len(uniqueMolecules))/float(len(activeSet) or 1)

    def OcurrencesIn(self, molecule):    
        pattern=OBSmartsPattern()
        pattern.Init(self.ToSmiles())
        return pattern.Match(molecule)
    
    def ContainedIn(self, molecule):
        return bool(self.OcurrencesIn(molecule))
            
class FragmentMiner(object):
    """A class for finding frequent molecular fragments
    Attributes:
        active      : list of smiles codes of active molecules
        inactive    : list of smiles codes of inactive molecules
        minSupport  : minimum frequency in the active set of the fragments to search for
        maxSupport  : maximum frequency in the inactive set of the fragments to search for
        addWholeRings : if True rings will be added as a whole rather then atom by atom
        canonicalPruning : if True a cache of all cannonical codes of all fragments will be kept to avoid
                    redundant search
        findClosed  : finds only fragments that are not sub-structures of any other fragment with the same support (default: True)
    Example:
    >>> miner = FragmentMiner(active = ["CC(C=N)=O", "c1ccccc1C=O", "SCC(N)O"], inactive = [], minSupport = 0.6)
    >>> for fragment in miner.Search():
    ... 	print fragment.ToSmiles() , "Support: %.3f" %fragment.Support()
    """
    loader=OBConversion()
    loader.SetInAndOutFormats("smi","smi")
    def __init__(self, active, inactive=[], minSupport=0.2, maxSupport=0.2, addWholeRings=True, canonicalPruning=True, findClosed=True):
        self.active=filter(lambda m:m, map(self.LoadMolecules, active))
        self.inactive=filter(lambda m:m, map(self.LoadMolecules, inactive))
        self.minSupport=minSupport
        self.maxSupport=maxSupport
        self.rings={}
        self.atomCount={}
        self.findClosed=findClosed
        self.addWholeRings=addWholeRings
        self.canonicalPruning=canonicalPruning
        self.canonicalPruningSet={}

    def LoadMolecules(self, smiles):
        mol=LoadMolFromSmiles(smiles)
        if mol:
            mol.StripSalts()
        return mol
        
    def GetAllMolecules(self):
        return self.active+self.inactive     

    def Initialize(self):
        """Initializes the search"""
        self.initialFragments=[]
        self.rings={}
        self.atomCount={}
        self.canonicalPruningSet={}
        candidates=[]
        ringCandidates=[]
        for mol in self.GetAllMolecules():
            mol.rings=self.rings[mol]=list(mol.GetSSSR())
            for ring in mol.rings:
                ring.fingerprint=set([mol.GetAtom(i).GetAtomicNum() for i in ring._path])
##            for ring in mol.rings:
##                candidates.append(FragmentExtensionByRing([ring],[Embeding(molecule=mol)]))
        
        for mol in self.GetAllMolecules():
            for atom in OBMolAtomIter(mol):
                self.atomCount[atom.GetAtomicNum()]= self.atomCount[atom.GetAtomicNum()]+1 if  atom.GetAtomicNum() in self.atomCount else 1
                if not (atom.GetAtomicNum()==6 and atom.IsInRing()):
                    candidates.append(FragmentExtensionByAtom(atom.GetAtomicNum(), atom.IsAromatic(), [atom.GetIdx()], [Embeding(molecule=mol)]))
        groups=[]
        candidates.sort(lambda a,b: cmp(self.atomCount[a.atomicNum], self.atomCount[b.atomicNum]))
        for extension in candidates:
            for ext in groups:
                if type(ext)==type(extension) and ext.IsEquivalent(extension):
                    ext.MergeFrom(extension)
                    break
            else:
                groups.append(extension)
        lst=self.atomCount.items()
        lst.sort(lambda a,b:cmp(a[1], b[1]))
        self.initialFragments=[]
        lst=[t[0] for t in lst]
        for extension in groups:
            if type(extension)==FragmentExtensionByAtom:
                f=Fragment(miner=self, excludeAtomList=lst[:lst.index(extension.atomicNum)])
            else:
                f=Fragment(miner=self)
            extension.Extend(f)
            self.initialFragments.append(f)
##        self.initialFragments.reverse()
##        excludeList=[]
##        for atom, c in lst:
##            f=Fragment(miner=self, excludeAtomList=excludeList)
##            f.InitializeFragment(atom)
##            self.initialFragments.append(f)
##            excludeList.append(atom)
        self.activeSet=set(self.active)
        self.inactiveSet=set(self.inactive)
            
    def TraverseTree(self, fragment):
        if self.canonicalPruning:
            codeWord=fragment.ToCannonicalSmiles()
            if codeWord in self.canonicalPruningSet:
                return self.canonicalPruningSet[codeWord]
            else:
                self.canonicalPruningSet[codeWord]=fragment.Support(self.activeSet)
        extended=fragment.Extend()
        extended=filter(lambda f:f.Support(self.activeSet)>=self.minSupport, extended)
        superStructSupport=[]
        for frag in extended:
            #print self.loader.WriteString(frag.ToOBMol())
            superStructSupport.append(self.TraverseTree(frag))
        support=fragment.Support(self.activeSet)
        if support>=self.minSupport and fragment.Support(self.inactiveSet)<=self.maxSupport:
            if not self.findClosed or (support not in superStructSupport):
                print fragment.ToSmiles().strip()+" %.2f %.2f" % (support, fragment.Support(self.inactiveSet))
                self.foundFragments.append(fragment)
        return support

    def Search(self):
        """Runs the search and returns the found fragments"""
        self.Initialize()
##        set_trace()
        self.foundFragments=[]
        for fragment in self.initialFragments:
            self.TraverseTree(fragment)
        #self.foundFragments=filter(lambda f:f.Support(self.inactive)<=self.maxSupport, self.foundFragments)
        return self.foundFragments

    def TraverseTreeIterator(self, fragment):
        if self.canonicalPruning:
            codeWord=fragment.ToCannonicalSmiles()
            if codeWord in self.canonicalPruningSet:
                raise StopIteration
            else:
                self.canonicalPruningSet[codeWord]=fragment.Support(self.activeSet)
        extended=fragment.Extend()
        extended=filter(lambda f:f.Support(self.activeSet)>=self.minSupport, extended)
        superStructSupport=[]
        for frag in extended:
            #print self.loader.WriteString(frag.ToOBMol())
            iter=self.TraverseTreeIterator(frag)
            try:
                while True:
                    f=iter.next()
                    superStructSupport.append(f.Support(self.active))
                    yield f
            except StopIteration:
                pass
                
            superStructSupport.append(self.TraverseTree(frag))
        support=fragment.Support(self.activeSet)
        if support>=self.minSupport and fragment.Support(self.inactiveSet)<=self.maxSupport:
            if not self.findClosed or (support not in superStructSupport):
                #print fragment.ToSmiles().strip()+" %.2f %.2f" % (support, fragment.Support(self.inactiveSet))
                #self.foundFragments.append(fragment)
                yield fragment
        #return support

    def SearchIterator(self):
        """Runs the search and returns the found fragments one by one"""
        self.Initialize()
##        set_trace()
        self.foundFragments=[]
        for fragment in self.initialFragments:
            iter=self.TraverseTreeIterator(fragment)
            try:
                while True:
                    yield iter.next()
            except StopIteration:
                pass
        #self.foundFragments=filter(lambda f:f.Support(self.inactive)<=self.maxSupport, self.foundFragments)
        #return self.foundFragments    
    
def LoadMolFromSmiles(smiles):
    """Returns an OBMol construcetd from an SMILES code"""
    mol=OBMol()
    loader=OBConversion()
    loader.SetInAndOutFormats("smi","smi")
    if not loader.ReadString(mol, smiles):
        return None
    mol.smilesCode=smiles
    return mol
    
class Fragmenter(object):
    """An object that is used to fragment an ExampleTable
    Attributes:
        minSupport  : minimum frequency in the active set of the fragments to search for (default: 0.2)
        maxSupport  : maximum frequency in the inactive set of the fragments to search for (default: 0.2)
        findClosed  : finds only fragments that are not sub-structures of any other fragment with the same support (default: True)
    Example:
    >>> fragmenter=Fragmenter(minSupport=0.1, maxSupport=0.05)
    >>> data, fragments=fragmenter(data, "SMILES", lambda ex:ex.getclass())
    """
    def __init__(self, minSupport=0.2, maxSupport=0.2, canonicalPruning=True, findClosed=True):
        self.minSupport=minSupport
        self.maxSupport=maxSupport
        self.canonicalPruning=canonicalPruning
        self.findClosed=findClosed
    def __call__(self, data, smilesAttr=None, activeFunc=lambda e:True):
        """Takes a data-set, and runs the FragmentMiner on it. Returns a new data-set and the fragments.
        The new data-set contains new attributes that represent the presence of a fragment that was found.
        Arguments:
            data        : the dataset
            smilesAttr  : the attribute in the data that contains the SMILES codes
            activeFunc  : a function that takes an example from the data-set and returns True if the example should be
                    considered as active (if none is provided all examples are considered active)
        """
        if not smilesAttr:
            smilesAttr=self.FindSmilesAttr(data)
        active=filter(lambda s:s, [str(e[smilesAttr]) for e in data if activeFunc(e)])
        inactive=filter(lambda s:s, [str(e[smilesAttr]) for e in data if not activeFunc(e)])
        
        miner=FragmentMiner(active, inactive, self.minSupport, self.maxSupport, canonicalPruning=self.canonicalPruning, findClosed=self.findClosed)
        self.fragments=fragments=miner.Search()
        fragVars=[orange.FloatVariable(frag.ToSmiles(), numberOfDecimals=0) for frag in fragments]
        smilesInFragments=dict([(fragment, set([embeding.molecule.smilesCode for embeding in fragment.embedings]) ) for fragment in fragments])
        from functools import partial
        def getVal(var, fragment, smilesAttr, example, returnWhat):
            mol=LoadMolFromSmiles(str(example[smilesAttr]))
##            print "GetVal"
            return fragment.ContainedIn(mol) and var(1) or var(0) if mol else None
        for var, frag in zip(fragVars, fragments):
            var.getValueFrom=partial(getVal,var, frag, smilesAttr)
        vars=data.domain.attributes+fragVars+(data.domain.classVar and [data.domain.classVar] or [])
        domain=orange.Domain(vars, data.domain.classVar and 1 or 0)
        domain.addmetas(data.domain.getmetas())
        table=orange.ExampleTable(domain)
        for e in data:
            vals=[e[attr] for attr in data.domain.attributes]+[1 if str(e[smilesAttr]) in smilesInFragments[fragment] else 0 for fragment in fragments]
            vals=vals + [e.getclass()] if data.domain.classVar else vals
            ex=orange.Example(domain, vals)
            for key, val in e.getmetas().items():
                ex[key]=val
            table.append(ex)
        return table, fragments
    def FindSmilesAttr(self, data):
        data=data.select(orange.MakeRandomIndices2(data, min(20, len(data))))
        stringVars=filter(lambda var:type(var)==orange.StringVariable, data.domain.attributes+data.domain.getmetas().values())
        count=dict.fromkeys(stringVars, 0)
        for example in data:
            for var in stringVars:
                if LoadMolFromSmiles(str(example[var])):
                    count[var]+=1
        count=count.items()
        count.sort(lambda a,b:cmp(a[1], b[1]))
        return count[-1][0]
            
import orngSVM
class FragmentBasedLearner(orange.Learner):
    """A learner wrapper class that first runs the molecular fragmentation on the data.
    Attributes:
        smilesAttr  : Attribute in the data that contains the smiles codes (if none is provided it will try to make a smart guess)
        learner     : learner that will be used to actualy learn on the fragmented data (default: orngSVM.SVMLearner)
        minSupport  : minimum frequency in the active set of the fragments to search for
        maxSupport  : maximum frequency in the inactive set of the fragments to search for
        activeFunc  : a function that takes an example from the learning data-set and returns True if the example should be
                    considered as active (if none is provided all examples are considered active)
        findClosed  : finds only fragments that are not sub-structures of any other fragment with the same support (default: True)
    """
    def __new__(cls, data=None, weights=0, **kwds):
        learner=orange.Learner.__new__(cls, **kwds)
        if data:
            learner.__init__(**kwds)
            return learner(data)
        else:
            return learner
    def __init__(self, learner=orngSVM.SVMLearner(probability=True), name="FragmentBasedLearner",
                 minSupport=0.2, maxSupport=0.2, smilesAttr=None, findClosed=True, activeFunc=lambda e:True):
        self.name=name
        self.learner=learner
        self.minSupport=minSupport
        self.smilesAttr=smilesAttr
        self.activeFunc=activeFunc
        self.maxSupport=maxSupport
        self.findClosed=findClosed
    def __call__(self, data, weight=0):
        fragmenter=Fragmenter(minSupport=self.minSupport, maxSupport=self.maxSupport, findClosed=self.findClosed)
        data, fragments=fragmenter(data, self.smilesAttr, self.activeFunc)
        return FragmentBasedClassifier(self.learner(data), data.domain)

class FragmentBasedClassifier(object):
    def __init__(self, classifier, domain):
        self.classifier=classifier
        self.domain=domain
    def __call__(self, example, getBoth=orange.GetValue):
        example=orange.Example(self.domain, example)
        return self.classifier(example, getBoth)

def Count(smiles, fragment):
    mols=filter(lambda m:m, map(LoadMolFromSmiles, smiles))
    for mol in mols: mol.StripSalts()
    pattern=OBSmartsPattern()
    pattern.Init(fragment)
    return len(filter(lambda m:pattern.Match(m, True), mols))

def ContaindIn(smiles, fragment):
    mol=LoadMolFromSmiles(smiles)
    pattern=OBSmartsPattern()
    pattern.Init(fragment)
    return bool(pattern.Match(mol))
    
def test():
    import orange
    d=orange.ExampleTable("E:\chem\mutagen_raw.tab")
    active=[str(e["SMILES"]) for e in d if str(e[-1])=="1"]
    inactive=[str(e["SMILES"]) for e in d if str(e[-1])=="0"]
##    d=orange.ExampleTable("E:\PCLedit_s.tab")
##    active=[str(e["SMILES"]) for e in d if not e["SMILES"].isSpecial()][:100]
##    print active
##    inactive=[]
##    active=["NC(C)C(=O)O", "NC(CS)C(=O)O", "NC(CO)C(=O)O"]
##    active=["CCS(O)(O)N", "CCS(O)(C)N", "CS(O)(C)N", "CCS(=N)N", "CS(=N)N", "CS(=N)O"]
##    active=["NC(S)c1ccccc1","NCC1=CC=CC=C1", "NCC1C=CC=CC=1", "c1ccccc1C(N)C(=S(O)C)c2ccccc2"]
##    active=["c1ccccc1C(N)C(=S(O)C)c2ccccc2"]
##    active=["O=C1C=CC(=O)C=C1","O=C1CCCCCN1"]
##    active=["C1SC2CCN2C1C(=O)"]
##    active=["CCCCCCc1ccc(O)cc1O","Nc1ccc(O)c(N)c1", "Cc1cc(C)c(N)cc1C", "CN(C)C(=S)S[Zn]SC(=S)N(C)C", "NC(=O)N(CCO)N=O"]
##    active=["CC(C)CCCC(C)C1CCC2C3CC=C4CC(CCC4(C)C3CCC12C)OC(=O)Cc5ccc(cc5)N(CCCl)CCCl","Cc1cc(C)c(N=Nc2c(O)c(cc3cc(ccc23)S(=O)(=O)O)S(=O)(=O)O)cc1CCNNCc1ccc(cc1)C(=O)NC(C)C","CC(C)(Oc1ccc(cc1)C2CCCc3ccccc23)C(=O)O","CCn1cc(C(=O)O)c(=O)c2ccc(C)nc12"]
    active=["CN(C)CCCN1c2ccccc2Sc3c1cc(cc3)C(F)(F)F", "CN(C)CCCN1c2ccccc2Sc3c1cc(cc3)Cl","CN1CCCCC1CCN2c3ccccc3Sc4c2cc(cc4)SC","c1ccc2c(c1)N(c3cc(ccc3S2)Cl)CCCN4CCN(CC4)CCO",
            "CN1CCN(CC1)CCCN2c3ccccc3Sc4c2cc(cc4)S(=O)(=O)N(C)C", "CN1CCC(=C2c3ccccc3Sc4c2cccc4)CC1", "[U]-C-S-P"]
##    active=["CN(C)CCCN1c2ccccc2Sc3c1cc(cc3)C(F)(F)F", "CN(C)CCCN1c2ccccc2Sc3c1cc(cc3)Cl","CN1CCCCC1CCN2c3ccccc3Sc4c2cc(cc4)SC"]
##    active=["CN(C)CCCN1c2ccccc2Sc3c1cc(cc3)C(F)(F)F", "CN(C)CCCN1c2ccccc2Sc3c1cc(cc3)Cl"]
##    set_trace()
    miner=FragmentMiner(active, inactive[:0], minSupport=0.1, maxSupport=0.1, addWholeRings=True, canonicalPruning=True)
    fragments=miner.Search()
##    for f in fragments:
##        print f.ToSmiles()

def test1():
    import orange
    data=orange.ExampleTable("E:\chem\mutagen_raw.tab")
##    data=orange.ExampleTable("E:\chem\smiles.tab")
    fragmenter=Fragmenter(minSupport=0.02, maxSupport=0.1, canonicalPruning=True)
##    set_trace()
    data, fragments1=fragmenter(data, "SMILES") #, lambda e:str(e[-1])=="1")
##    data, fragments2=fragmenter(data, "SMILES", lambda e:str(e[-1])=="0")
    data.save("E:\chem\mutagen_raw_frag.tab")
    
if __name__=="__main__":
    import time
    sTime=time.clock()
    test1()
    print time.clock()-sTime

