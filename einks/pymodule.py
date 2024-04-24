#
# @BEGIN LICENSE
#
# einks by Psi4 Developer, a plugin to:
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

def run_einks(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    einks can be called via :py:func:`~driver.energy`. For scf plugins.

    >>> energy('einks')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    # Build a new blank wavefunction to pass into scf
    einks_molecule = kwargs.get('molecule', psi4.core.get_active_molecule())

    func_str = "b3lyp"

    if "dft_functional" in kwargs :
        func_str = kwargs["dft_functional"]
        
    func, disp_type = psi4.dft.build_superfunctional(func_str, False)

    ref_wfn = psi4.core.Wavefunction.build(einks_molecule, psi4.core.get_global_option('BASIS'))

    new_wfn = psi4.proc.scf_wavefunction_factory(func_str, ref_wfn, "RKS", **kwargs)

    einks_wfn = psi4.core.plugin('einks.so', new_wfn)
    psi4.set_variable('CURRENT ENERGY', einks_wfn.energy())

    if kwargs.get('ref_wfn', False):
        return (einks_wfn, einks_wfn.energy())
    else:
        return einks_wfn.energy()

# Integration with driver routines
psi4.driver.procedures['energy']['einks'] = run_einks


def exampleFN():
    # Your Python code goes here
    pass
