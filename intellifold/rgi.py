"""Extract IntelliFold input chemistry for the shared RGI adapter."""

import logging

from alphafold3.common import folding_input
from alphafold3.constants import chemical_components, residue_names
from alphafold3.data.tools import rdkit_utils
from rdkit import Chem

from rgi_toolkit.combined import CombinedRestraints
from rgi_toolkit.intellifold.adapter import IntelliFoldAdapter
from rgi_toolkit.optim.scan_runner import ScanMinimizer


def _ccd_ligand_mol(ccd_cif):
    """Keep valid CCD graphs when ideal coordinates are unavailable."""
    options = {"sort_alphabetically": False, "remove_hydrogens": True}
    try:
        return rdkit_utils.mol_from_ccd_cif(ccd_cif, **options)
    except rdkit_utils.MolFromMmcifError:
        keys = [f"_chem_comp_atom.pdbx_model_Cartn_{axis}_ideal" for axis in "xyz"]
        if not any(
            key not in ccd_cif or any(value in {"?", "."} for value in ccd_cif[key])
            for key in keys
        ):
            raise
        # Relax coordinates only; malformed atom and bond data must still fail.
        try:
            rdkit_utils.parse_atom_data(ccd_cif, force_parse=False)
            rdkit_utils.parse_bond_data(ccd_cif, force_parse=False)
        except (KeyError, ValueError) as error:
            raise rdkit_utils.MolFromMmcifError from error
        mol = rdkit_utils.mol_from_ccd_cif(ccd_cif, force_parse=True, **options)
        logging.getLogger(__name__).warning(
            "CCD %s has no complete ideal coordinates; using its atom/bond graph "
            "without stereochemistry inferred from ideal coordinates",
            ccd_cif.get("_chem_comp.id", ["unknown"])[0],
        )
        return mol


def build_restraints(fold_input, example):
    """Build fresh restraint state from this structure and this seed's features."""
    if fold_input.restraints_config is None:
        return None
    chain_id_to_asym = {chain.id: i + 1 for i, chain in enumerate(fold_input.chains)}
    conformer_by_asym = {
        chain_id_to_asym[chain.id]: chain.conformer_restraints
        for chain in fold_input.chains
    }
    ligand_mols = []
    ccd = None
    for chain in fold_input.chains:
        if (
            not isinstance(chain, folding_input.Ligand)
            or not chain.conformer_restraints
        ):
            continue
        if chain.smiles is not None:
            mol = Chem.MolFromSmiles(chain.smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES for restrained ligand {chain.id}")
            ligand_mols.append((chain.id, mol, True))
        elif chain.ccd_ids is not None and len(chain.ccd_ids) == 1:
            if ccd is None:
                ccd = chemical_components.Ccd(user_ccd=fold_input.user_ccd)
            ccd_cif = ccd.get(chain.ccd_ids[0])
            if ccd_cif is None:
                raise ValueError(f"Missing CCD for restrained ligand {chain.id}")
            mol = _ccd_ligand_mol(ccd_cif)
            ligand_mols.append((chain.id, mol, False))
        else:
            raise ValueError(
                "Conformer restraints require a single-CCD or SMILES ligand"
            )
    adapter = IntelliFoldAdapter(
        batch=example,
        chain_id_to_asym=chain_id_to_asym,
        polymer_residue_names=residue_names.POLYMER_TYPES,
        ligand_mols=ligand_mols,
        conformer_restraints_by_asym=conformer_by_asym,
    )
    rgi = CombinedRestraints()
    rgi.setup(adapter, config=fold_input.restraints_config)
    return ScanMinimizer(rgi, rgi.get_minimizer() if rgi.is_active() else None)
