from openeye.oechem import *
from openeye.oedepict import *

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

if __name__=="__main__":
    import sys
    if len(sys.argv)!=3:
        molSmiles="CN(C)CCCN1C2=CC=CC=C2SC3=C1C=C(C=C3)Cl.Cl"
        fragSmiles="C-N"
    else:
        molSmiles=sys.argv[1]
        fragSmiles=sys.argv[2]
    molecule2BMP(molSmiles, "mol.bmp")
    moleculeFragment2BMP(molSmiles, fragSmiles, "mol_sub.bmp")
