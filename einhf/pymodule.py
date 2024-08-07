#
# @BEGIN LICENSE
#
# einhf by Psi4 Developer, a plugin to:
#
# Psi4: an open-source quantum chemistry software package
#
# Copyright (c) 2007-2023 The Psi4 Developers.
#
# The copyrights for code used from other parties are included in
# the corresponding files.
#
# This file is part of Psi4.
#
# Psi4 is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3.
#
# Psi4 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with Psi4; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# @END LICENSE
#

import psi4
import psi4.driver.p4util as p4util
from psi4.driver.procrouting import proc_util

def run_einhf(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    einhf can be called via :py:func:`~driver.energy`. For scf plugins.

    >>> energy('einhf')

    """

    print(f"Plugin load status: {psi4.core.plugin_load('einhf.so')}")
    
    psi4.core.reopen_outfile()
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    # Your plugin's psi4 run sequence goes here
    #psi4.core.set_local_option('EINHF', 'PRINT', 1)

    # Build a new blank wavefunction to pass into scf
    einhf_molecule = kwargs.get('molecule', psi4.core.get_active_molecule())

    aux_basis = psi4.core.BasisSet.build(einhf_molecule, key = "DF_BASIS_SCF", target = psi4.core.get_option("SCF", "DF_BASIS_SCF"),
                                         fitrole = "JKFIT", other = psi4.core.get_global_option("BASIS"))
    func_str = "hf"

    if "dft_functional" in kwargs :
        func_str = kwargs["dft_functional"]

    timer_str = "EinHF: " + psi4.core.get_global_option("COMPUTE").upper() + " "

    if psi4.core.get_global_option("REFERENCE").lower() == "rhf" and func_str.lower() != "hf" :
        timer_str = timer_str + "RKS"
    elif psi4.core.get_global_option("REFERENCE").lower() == "uhf" and func_str.lower() != "hf" :
        timer_str = timer_str + "UKS"
    else :
        timer_str = timer_str + psi4.core.get_global_option("REFERENCE")

    psi4.core.timer_on(timer_str)
        
    #func, disp_type = psi4.dft.build_superfunctional(func_str, False)

    ref_wfn = psi4.core.Wavefunction.build(einhf_molecule, psi4.core.get_global_option('BASIS'))

    new_wfn = psi4.proc.scf_wavefunction_factory(func_str, ref_wfn, psi4.core.get_global_option("REFERENCE"), **kwargs)
    
    new_wfn.set_basisset("DF_BASIS_SCF", aux_basis)

    einhf_wfn = psi4.core.plugin('einhf.so', new_wfn)
    psi4.set_variable('CURRENT ENERGY', einhf_wfn.energy())

    psi4.core.timer_off(timer_str)

    if kwargs.get('ref_wfn', False):
        return (einhf_wfn, einhf_wfn.energy())
    else:
        return einhf_wfn.energy()

# Integration with driver routines
psi4.driver.procedures['energy']['einhf'] = run_einhf


def exampleFN():
    # Your Python code goes here
    pass
