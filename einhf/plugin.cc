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
#include <memory>

#ifdef __HIP__
#include "rhf-gpu.h"
#include "uhf-gpu.h"
#endif

#include "rmp2.h"
#include "ump2.h"

#ifdef __HIP__
#include "rmp2-gpu.h"
#include "ump2-gpu.h"
#endif

#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libqt/qt.h"
#include "psi4/libscf_solver/rhf.h"
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
    // options.add_int("PRINT", 1);
    // /*- How tightly to converge the energy -*/
    // options.add_double("E_CONVERGENCE", 1.0E-10);
    // /*- How tightly to converge the density -*/
    // options.add_double("D_CONVERGENCE", 1.0E-6);
    // /*- How many iteration to allow -*/
    // options.add_int("MAXITER", 50);
    // /*- Whether to use DIIS acceleration. -*/
    // options.add_bool("DIIS", true);
    // /*- How many DIIS vectors to store. -*/
    // options.add_int("DIIS_MAX_VECS", 6);
    // /*- What reference to use -*/
    // options.add_str("REFERENCE", "RHF");
    /*- What kind of SCF to do. -*/
    options.add_str("COMPUTE", "CPU");
    /*- What computation to do. -*/
    options.add_str("METHOD", "SCF");
    /*- Output file name. -*/
    options.add_array("OUTFILE");
  }

  return true;
}

extern "C" PSI_API SharedWavefunction einhf(SharedWavefunction ref_wfn,
                                            Options &options) {

  omp_set_num_threads(Process::environment.get_n_threads());

  std::string timer_str = "EinHF: " + options.get_str("COMPUTE") + " " +
                          options.get_str("REFERENCE");
  std::shared_ptr<EinsumsRHF> rhfwfn;
  std::shared_ptr<EinsumsUHF> uhfwfn;
  std::shared_ptr<EinsumsRMP2> rmp2wfn;
  std::shared_ptr<EinsumsUMP2> ump2wfn;
#ifdef BUILD_GPU
  std::shared_ptr<GPUEinsumsRHF> gpu_rhfwfn;
  std::shared_ptr<GPUEinsumsUHF> gpu_uhfwfn;
  std::shared_ptr<GPUEinsumsRMP2> gpu_rmp2wfn;
  std::shared_ptr<GPUEinsumsUMP2> gpu_ump2wfn;
#endif

  SharedWavefunction outwfn;

  std::shared_ptr<Functional> functional;

  if (!psi::outfile) {
    printf("No output file.\n");
  }

  psi::outfile->Printf("Initializing Einsums.\n");
  einsums::initialize();
  if ((to_lower(options.get_str("REFERENCE")) == "rhf" ||
       to_lower(options.get_str("REFERENCE")) == "rks") &&
      to_lower(options.get_str("COMPUTE")) == "cpu") {
    // Build an SCF object, and tell it to compute its energy
    rhfwfn = std::make_shared<EinsumsRHF>(
        ref_wfn, static_cast<psi::scf::HF *>(ref_wfn.get())->functional(),
        options);
#pragma omp parallel
    {
#pragma omp single
      { rhfwfn->compute_energy(); }
    }
    outwfn = std::static_pointer_cast<Wavefunction>(rhfwfn);

  } else if ((to_lower(options.get_str("REFERENCE")) == "uhf" ||
              to_lower(options.get_str("REFERENCE")) == "uks") &&
             to_lower(options.get_str("COMPUTE")) == "cpu") {
    // Build an SCF object, and tell it to compute its energy
    uhfwfn = std::make_shared<EinsumsUHF>(
        ref_wfn, static_cast<psi::scf::HF *>(ref_wfn.get())->functional(),
        options);
#pragma omp parallel
    {
#pragma omp single
      { uhfwfn->compute_energy(); }
    }
    outwfn = std::static_pointer_cast<Wavefunction>(uhfwfn);
#ifdef BUILD_GPU
  } else if ((to_lower(options.get_str("REFERENCE")) == "rhf" ||
              to_lower(options.get_str("REFERENCE")) == "rks") &&
             to_lower(options.get_str("COMPUTE")) == "gpu") {
    // Build an SCF object, and tell it to compute its energy
    gpu_rhfwfn = std::make_shared<GPUEinsumsRHF>(
        ref_wfn, static_cast<psi::scf::HF *>(ref_wfn.get())->functional(),
        options);
#pragma omp parallel
    {
#pragma omp single
      { gpu_rhfwfn->compute_energy(); }
    }
    outwfn = std::static_pointer_cast<Wavefunction>(gpu_rhfwfn);
  } else if ((to_lower(options.get_str("REFERENCE")) == "uhf" ||
              to_lower(options.get_str("REFERENCE")) == "uks") &&
             to_lower(options.get_str("COMPUTE")) == "gpu") {
    // Build an SCF object, and tell it to compute its energy
    gpu_uhfwfn = std::make_shared<GPUEinsumsUHF>(
        ref_wfn, static_cast<psi::scf::HF *>(ref_wfn.get())->functional(),
        options);
#pragma omp parallel
    {
#pragma omp single
      { gpu_uhfwfn->compute_energy(); }
    }
    outwfn = std::static_pointer_cast<Wavefunction>(gpu_uhfwfn);
#endif
  } else {
    throw PSIEXCEPTION("Unable to handle reference" +
                       options.get_str("REFERENCE"));
  }

  if (to_lower(options.get_str("METHOD")) == "mp2" ||
      to_lower(options.get_str("METHOD")) == "ccsd") {
    if (to_lower(options.get_str("REFERENCE")) == "rhf" &&
        to_lower(options.get_str("COMPUTE")) == "cpu") {
      rmp2wfn = std::make_shared<EinsumsRMP2>(rhfwfn, options);
#pragma omp parallel
      {
#pragma omp single
        { rmp2wfn->compute_energy(); }
      }
      outwfn = std::static_pointer_cast<Wavefunction>(rmp2wfn);
    } else if (to_lower(options.get_str("REFERENCE")) == "uhf" &&
               to_lower(options.get_str("COMPUTE")) == "cpu") {
      ump2wfn = std::make_shared<EinsumsUMP2>(uhfwfn, options);
#pragma omp parallel
      {
#pragma omp single
        { ump2wfn->compute_energy(); }
      }
      outwfn = std::static_pointer_cast<Wavefunction>(ump2wfn);
#ifdef __HIP__
    } else if (to_lower(options.get_str("REFERENCE")) == "rhf" &&
               to_lower(options.get_str("COMPUTE")) == "gpu") {
      gpu_rmp2wfn = std::make_shared<GPUEinsumsRMP2>(gpu_rhfwfn, options);
#pragma omp parallel
      {
#pragma omp single
        { gpu_rmp2wfn->compute_energy(); }
      }
      outwfn = std::static_pointer_cast<Wavefunction>(gpu_rmp2wfn);          
    } else if (to_lower(options.get_str("REFERENCE")) == "uhf" &&
               to_lower(options.get_str("COMPUTE")) == "gpu") {
      gpu_ump2wfn = std::make_shared<GPUEinsumsUMP2>(gpu_uhfwfn, options);
#pragma omp parallel
      {
#pragma omp single
        { gpu_ump2wfn->compute_energy(); }
      }
      outwfn = std::static_pointer_cast<Wavefunction>(gpu_ump2wfn);          
#endif
    } else {
      throw PSIEXCEPTION(
          "Can not handle MP2 with the given reference or compute type!");
    }
  } else if (to_lower(options.get_str("METHOD")) == "scf") {
    einsums::finalize(false);
    return outwfn;
  } else {
    throw PSIEXCEPTION("Unrecognized method" + options.get_str("METHOD"));
  }

  if (to_lower(options.get_str("METHOD")) == "mp2") {
    return outwfn;
  } else {
    throw PSIEXCEPTION("Unrecognized method" + options.get_str("METHOD"));
  }
}

} // namespace einhf
} // namespace psi
