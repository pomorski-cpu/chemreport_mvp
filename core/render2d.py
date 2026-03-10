from __future__ import annotations
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

def mol_to_svg(mol: Chem.Mol, width: int = 520, height: int = 420) -> str:
    # Подготовка 2D-координат (важно для стабильного вида)
    m = Chem.Mol(mol)
    if not m.GetNumConformers():
        Chem.rdDepictor.Compute2DCoords(m)

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()

    # Визуальная настройка
    opts.clearBackground = True           # прозрачный фон
    opts.padding = 0.08                   # поля вокруг структуры (0.0..0.2)
    opts.bondLineWidth = 1.6              # толщина связей (меньше = аккуратнее)
    opts.minFontSize = 12                 # шрифт атомов
    opts.maxFontSize = 22
    opts.fixedBondLength = 22             # стабилизирует масштаб (по желанию)
    opts.addAtomIndices = False
    opts.includeAtomTags = False

    # Рисуем
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, m)
    drawer.FinishDrawing()

    svg = drawer.GetDrawingText()

    # RDKit иногда вставляет xml header/лишние размеры; это ок для QSvgWidget.
    return svg
