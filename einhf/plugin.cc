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
 * You should have received a copy of the GNU Lesser General Public License
 * along with Psi4; if not, write to the Free Software Foundation, Inc., 51
 * Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#include "rhf.h"
#include "uhf.h"

#ifdef __HIP__
#include "rhf-gpu.h"
#include "uhf-gpu.h"
#endif

#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libqt/qt.h"
#include "psi4/psi4-dec.h"

static std::string to_lower(const std::string &str) {
  std::string out(str);
  std::transform(str.begin(), str.end(), out.begin(),
                 [](char c) { return std::tolower(c); });
  return out;
}

namespace psi {
namespace einhf {

extern "C" PSI_API int read_options(std::string name, Options &options) {
  if (name == "EINHF" || options.read_globals()) {
    /*- The amount of information printed
        to the output file -*/
    options.add_int("PRINT", 1);
    /*- How tightly to converge the energy -*/
    options.add_double("E_CONVERGENCE", 1.0E-10);
    /*- How tightly to converge the density -*/
    options.add_double("D_CONVERGENCE", 1.0E-6);
    /*- How many iteration to allow -*/
    options.add_int("SCF_MAXITER", 50);
    /*- Whether to use DIIS acceleration. -*/
    options.add_bool("DIIS", true);
    /*- How many DIIS vectors to store. -*/
    options.add_int("DIIS_MAX_VECS", 6);
    /*- What reference to use -*/
    options.add_str("REFERENCE", "RHF");
    /*- What kind of SCF to do. -*/
    options.add_str("COMPUTE", "CPU");
  }

  return true;
}

extern "C" PSI_API SharedWavefunction einhf(SharedWavefunction ref_wfn,
                                            Options &options) {
  timer_on("EinHF");

  omp_set_num_threads(Process::environment.get_n_threads());

  if (!outfile) {
    printf("No output file.\n");
  }
  outfile->Printf("Initializing Einsums.\n");
  einsums::initialize();
  if (to_lower(options.get_str("REFERENCE")) == "rhf" &&
      to_lower(options.get_str("COMPUTE")) == "cpu") {
    // Build an SCF object, and tell it to compute its energy
    SharedWavefunction scfwfn =
        std::shared_ptr<Wavefunction>(new EinsumsSCF(ref_wfn, options));
#pragma omp parallel
    {
#pragma omp single
      { scfwfn->compute_energy(); }
    }

    einsums::finalize(true);
    timer_off("EinHF");

    return scfwfn;
  } else if (to_lower(options.get_str("REFERENCE")) == "uhf" &&
             to_lower(options.get_str("COMPUTE")) == "cpu") {
    // Build an SCF object, and tell it to compute its energy
    SharedWavefunction scfwfn =
        std::shared_ptr<Wavefunction>(new EinsumsUHF(ref_wfn, options));
#pragma omp parallel
    {
#pragma omp single
      { scfwfn->compute_energy(); }
    }

    einsums::finalize(true);
    timer_off("EinHF");

    return scfwfn;
  } else if (to_lower(options.get_str("REFERENCE")) == "rhf" &&
             to_lower(options.get_str("COMPUTE")) == "gpu") {
    // Build an SCF object, and tell it to compute its energy
    SharedWavefunction scfwfn =
        std::shared_ptr<Wavefunction>(new GPUEinsumsSCF(ref_wfn, options));
#pragma omp parallel
    {
#pragma omp single
      { scfwfn->compute_energy(); }
    }

    einsums::finalize(true);
    timer_off("EinHF");

    return scfwfn;
  } else if (to_lower(options.get_str("REFERENCE")) == "uhf" &&
             to_lower(options.get_str("COMPUTE")) == "gpu") {
    // Build an SCF object, and tell it to compute its energy
    SharedWavefunction scfwfn =
        std::shared_ptr<Wavefunction>(new GPUEinsumsUHF(ref_wfn, options));
#pragma omp parallel
    {
#pragma omp single
      { scfwfn->compute_energy(); }
    }

    einsums::finalize(true);
    timer_off("EinHF");

    return scfwfn;
  } else {
    throw PSIEXCEPTION("Unable to handle reference" +
                       options.get_str("REFERENCE"));
  }
}

} // namespace einhf
} // namespace psi
