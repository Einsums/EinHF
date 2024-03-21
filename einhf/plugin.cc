/*
 * @BEGIN LICENSE
 *
 * einhf by Psi4 Developer, a plugin to:
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2023 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Psi4 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with Psi4; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#include "scf.h"

#include "psi4/psi4-dec.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"
#include "psi4/libpsi4util/PsiOutStream.h"

namespace psi{ namespace einhf {

extern "C" PSI_API
int read_options(std::string name, Options &options)
{
    if (name == "EINHF"|| options.read_globals()) {
        /*- The amount of information printed
            to the output file -*/
        options.add_int("PRINT", 1);
        /*- How tightly to converge the energy -*/
        options.add_double("E_CONVERGENCE", 1.0E-10);
        /*- How tightly to converge the density -*/
        options.add_double("D_CONVERGENCE", 1.0E-6);
        /*- How many iteration to allow -*/
        options.add_int("SCF_MAXITER", 50);
    }

    return true;
}

extern "C" PSI_API
SharedWavefunction einhf(SharedWavefunction ref_wfn, Options &options)
{
    einsums::initialize();
    // Build an SCF object, and tell it to compute its energy
    SharedWavefunction scfwfn = std::shared_ptr<Wavefunction>(new SCF(ref_wfn, options));
    scfwfn->compute_energy();

    einsums::finalize();

    return scfwfn;
}

}} // End Namespaces
