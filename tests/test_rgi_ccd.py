"""CCD recovery preserves atom identity without accepting malformed chemistry."""

import pytest
from alphafold3.constants import chemical_components
from alphafold3.data.tools import rdkit_utils
from rdkit import Chem

from intellifold.rgi import _ccd_ligand_mol


def signature(mol):
    atoms = [(a.GetProp("atom_name"), a.GetAtomicNum(), a.GetFormalCharge()) for a in mol.GetAtoms()]
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), str(b.GetBondType())) for b in mol.GetBonds()]
    return atoms, bonds, Chem.MolToSmiles(mol)


@pytest.mark.parametrize("code", ["AXT", "UUB", "MAU"])
def test_missing_ideal_coordinates_keep_native_ccd_graph(code, caplog):
    entry = chemical_components.Ccd()[code]
    with pytest.raises(rdkit_utils.MolFromMmcifError):
        rdkit_utils.mol_from_ccd_cif(entry)
    expected = rdkit_utils.mol_from_ccd_cif(
        entry, force_parse=True, sort_alphabetically=False, remove_hydrogens=True
    )
    actual = _ccd_ligand_mol(entry)
    assert actual.GetNumAtoms() > 0
    assert signature(actual) == signature(expected)
    assert "without stereochemistry inferred from ideal coordinates" in caplog.text


def test_complete_ideal_coordinates_preserve_stereo_and_conformer():
    entry = chemical_components.Ccd()["ATP"]
    expected = rdkit_utils.mol_from_ccd_cif(
        entry, sort_alphabetically=False, remove_hydrogens=True
    )
    actual = _ccd_ligand_mol(entry)
    assert signature(actual) == signature(expected)
    assert actual.GetNumConformers() == expected.GetNumConformers() == 1
    assert any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in actual.GetAtoms())


@pytest.mark.parametrize("field", ["_chem_comp_atom.charge", "_chem_comp_bond.pdbx_aromatic_flag"])
def test_missing_coordinates_do_not_relax_invalid_chemistry(field):
    entry = dict(chemical_components.Ccd()["AXT"])
    entry[field] = ["?", *entry[field][1:]]
    with pytest.raises((ValueError, rdkit_utils.MolFromMmcifError)):
        _ccd_ligand_mol(entry)


def test_invalid_coordinate_text_is_not_a_missing_coordinate():
    entry = dict(chemical_components.Ccd()["ATP"])
    key = "_chem_comp_atom.pdbx_model_Cartn_x_ideal"
    entry[key] = ["invalid", *entry[key][1:]]
    with pytest.raises(rdkit_utils.MolFromMmcifError):
        _ccd_ligand_mol(entry)
