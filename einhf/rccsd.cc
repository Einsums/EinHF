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
#include "rmp2.h"

#include "einsums.hpp"
#include "einsums/Tensor.hpp"

#include "psi4/libfock/jk.h"
#include "psi4/libfock/v.h"
#include "psi4/libfunctional/superfunctional.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/mintshelper.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/pointgrp.h"
#include "psi4/libmints/sobasis.h"
#include "psi4/libmints/sointegral_twobody.h"
#include "psi4/libmints/vector.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libqt/qt.h"
#include "psi4/psi4-dec.h"
#include <LinearAlgebra.hpp>
#include <_Common.hpp>
#include <_Index.hpp>
#include <cmath>

static std::string to_lower(const std::string &str) {
  std::string out(str);
  std::transform(str.begin(), str.end(), out.begin(),
                 [](char c) { return std::tolower(c); });
  return out;
}

using namespace einsums;
using namespace einsums::tensor_algebra;

namespace psi {
namespace einhf {

EinsumsRCCSD::EinsumsRCCSD(std::shared_ptr<EinsumsRHF> ref_wfn,
                           Options &options)
    : Wavefunction(options) {

  timer_on("EinHF: Setup RCCSD wavefunction");

  // Shallow copy useful objects from the passed in wavefunction
  shallow_copy(ref_wfn);

  mp2_wfn_ = ref_wfn;

  energy_ = ref_wfn->getSCFEnergy();

  e_convergence_ = options_.get_double("E_CONVERGENCE");
  d_convergence_ = options_.get_double("D_CONVERGENCE");
  maxiter_ = options_.get_int("CC_MAXITER");

  print_ = options_.get_int("PRINT");

  nirrep_ = sobasisset_->nirrep();
  nso_ = basisset_->nbf();
  occ_per_irrep_ = ref_wfn->getOccPerIrrep();
  irrep_sizes_ = ref_wfn->getIrrepSizes();
  unocc_per_irrep_ = std::vector<int>(nirrep_);
  irrep_offsets_ = std::vector<int>(nirrep_);
  irrep_offsets_[0] = 0;

  for (int i = 0; i < nirrep_; i++) {
    unocc_per_irrep_[i] = irrep_sizes_.at(i) - occ_per_irrep_.at(i);
    if (i != 0) {
      irrep_offsets_[i] = irrep_offsets_[i - 1] + irrep_sizes_.at(i - 1);
    }
  }

  print_header();

  init_integrals();

  timer_off("EinHF: Setup MP2 wavefunction");
}

EinsumsRCCSD::~EinsumsRCCSD() {}

void EinsumsRCCSD::init_integrals() {
  tei_anti_oooo =
      TiledTensor<double, 2>("Antisymmetrized TEI <OO||OO>", occ_per_irrep_,
                             occ_per_irrep_, occ_per_irrep_, occ_per_irrep_);
  tei_anti_ooov =
      TiledTensor<double, 2>("Antisymmetrized TEI <OO||OV>", occ_per_irrep_,
                             occ_per_irrep_, occ_per_irrep_, unocc_per_irrep_);
  tei_anti_oovv = TiledTensor<double, 2>("Antisymmetrized TEI <OO||VV>",
                                         occ_per_irrep_, occ_per_irrep_,
                                         unocc_per_irrep_, unocc_per_irrep_);
  tei_anti_ovov = TiledTensor<double, 2>("Antisymmetrized TEI <OV||OV>",
                                         occ_per_irrep_, unocc_per_irrep_,
                                         occ_per_irrep_, unocc_per_irrep_);
  tei_anti_ovvv = TiledTensor<double, 2>("Antisymmetrized TEI <OV||VV>",
                                         occ_per_irrep_, unocc_per_irrep_,
                                         unocc_per_irrep_, unocc_per_irrep_);
  tei_anti_vvvv = TiledTensor<double, 2>("Antisymmetrized TEI <VV||VV>",
                                         unocc_per_irrep_, unocc_per_irrep_,
                                         unocc_per_irrep_, unocc_per_irrep_);
  TiledTensor<double, 4> tei_anti("Antisymmetric TEI temp", irrep_sizes_);
  auto tei = mp2_wfn_->getTei();
  auto denominator = mp2_wfn_->getDenominator();

  // Antisymmetrize the two-electron integrals.
  einsums::tensor_algebra::sort(
      0.0, Indices{index::i, index::j, index::a, index::b}, &tei_anti, 1.0,
      Indices{index::i, index::a, index::j, index::b}, tei);
  einsums::tensor_algebra::sort(
      1.0, Indices{index::i, index::j, index::a, index::b}, &tei_anti, -1.0,
      Indices{index::i, index::b, index::j, index::a}, tei);

  for (int p = 0; p < nirrep_; p++) {
    for (int q = 0; q < nirrep_; q++) {
      for (int r = 0; r < nirrep_; r++) {
        for (int s = 0; s < nirrep_; s++) {
          tei_anti_oooo.tile(p, q, r, s) = tei_anti.tile(p, q, r, q)(
              Range{0, occ_per_irrep_[p]}, Range{0, occ_per_irrep_[q]},
              Range{0, occ_per_irrep_[r]}, Range{0, occ_per_irrep_[s]});
          tei_anti_ooov.tile(p, q, r, s) = tei_anti.tile(p, q, r, q)(
              Range{0, occ_per_irrep_[p]}, Range{0, occ_per_irrep_[q]},
              Range{0, occ_per_irrep_[r]},
              Range{occ_per_irrep_[s], irrep_sizes_[s]});
          tei_anti_oovv.tile(p, q, r, s) = tei_anti.tile(p, q, r, q)(
              Range{0, occ_per_irrep_[p]}, Range{0, occ_per_irrep_[q]},
              Range{occ_per_irrep_[r], irrep_sizes_[r]},
              Range{occ_per_irrep_[s], irrep_sizes_[s]});
          tei_anti_ovov.tile(p, q, r, s) = tei_anti.tile(p, q, r, q)(
              Range{0, occ_per_irrep_[p]},
              Range{occ_per_irrep_[q], irrep_sizes_[q]},
              Range{0, occ_per_irrep_[r]},
              Range{occ_per_irrep_[s], irrep_sizes_[s]});
          tei_anti_ovvv.tile(p, q, r, s) = tei_anti.tile(p, q, r, q)(
              Range{0, occ_per_irrep_[p]},
              Range{occ_per_irrep_[q], irrep_sizes_[q]},
              Range{occ_per_irrep_[r], irrep_sizes_[r]},
              Range{occ_per_irrep_[s], irrep_sizes_[s]});
          tei_anti_vvvv.tile(p, q, r, s) = tei_anti.tile(p, q, r, q)(
              Range{occ_per_irrep_[p], irrep_sizes_[p]},
              Range{occ_per_irrep_[q], irrep_sizes_[q]},
              Range{occ_per_irrep_[r], irrep_sizes_[r]},
              Range{occ_per_irrep_[s], irrep_sizes_[s]});
        }
      }
    }
  }
}

double EinsumsRCCSD::compute_energy() {
  timer_on("EinHF: Computing CCSD energy");

  double e_new;

  // Determine the number of electrons in the system
  // The molecule object is built into all wavefunctions
  int charge = molecule_->molecular_charge();
  int nelec_ = 0;
  for (int i = 0; i < molecule_->natom(); ++i) {
    nelec_ += (int)molecule_->Z(i);
  }
  nelec_ -= charge;
  if (nelec_ % 2) {
    throw PSIEXCEPTION("This is only an RCCSD code, but you gave it an odd "
                       "number of electrons.  Try again!");
  }

  timer_on("EinHF: Set up CCSD amplitudes.");

  t1_amps_ =
      TiledTensor<double, 2>("T1 amplitudes", occ_per_irrep_, unocc_per_irrep_);
  t2_amps_ =
      TiledTensor<double, 2>("T2 amplitudes", occ_per_irrep_, occ_per_irrep_,
                             unocc_per_irrep_, unocc_per_irrep_);

  // Set the initial T2 amplitudes.
  einsum(Indices{index::i, index::j, index::a, index::b}, &t2_amps_,
         Indices{index::i, index::j, index::a, index::b}, tei_anti_oovv,
         Indices{index::i, index::a, index::j, index::b}, denominator);

  // The initial T1 amplitudes are 0.
  // Set up the intermediates.
  TiledTensor<double, 2> Fae = TiledTensor<double, 2>(
      "F intermediate", unocc_per_irrep_, unocc_per_irrep_);
  TiledTensor<double, 4> Wmnij =
      TiledTensor<double, 4>("W intermediate", occ_per_irrep_, occ_per_irrep_,
                             occ_per_irrep_, occ_per_irrep_);
  TiledTensor<double, 2> Fmi =
      TiledTensor<double, 2>("F intermediate", occ_per_irrep_, occ_per_irrep_);
  TiledTensor<double, 4> Wabef = TiledTensor<double, 4>(
      "W intermediate", unocc_per_irrep_, unocc_per_irrep_, unocc_per_irrep_,
      unocc_per_irrep_);
  TiledTensor<double, 2> Fme = TiledTensor<double, 2>(
      "F intermediate", occ_per_irrep_, unocc_per_irrep_);
  TiledTensor<double, 4> Wmbej =
      TiledTensor<double, 4>("W intermediate", occ_per_irrep_, unocc_per_irrep_,
                             unocc_per_irrep_, occ_per_irrep_);
  TiledTensor<double, 4> tau =
      TiledTensor<double, 4>("tau intermediate", occ_per_irrep_, occ_per_irrep_,
                             unocc_per_irrep_, unocc_per_irrep_);
  TiledTensor<double, 4> tau_tilde = TiledTensor<double, 4>(
      "~tau intermediate", occ_per_irrep_, occ_per_irrep_, unocc_per_irrep_,
      unocc_per_irrep_);
  TiledTensor<double, 2> T1_prev = TiledTensor<double, 2>(
                             "Previous T1 amplitudes.", occ_per_irrep_,
                             unocc_per_irrep_),
                         T1_temp = TiledTensor<double, 2>(
                             "T1 amplitude temporary tensor.", occ_per_irrep_,
                             unocc_per_irrep_);
  TiledTensor<double, 4> T2_prev = TiledTensor<double, 4>(
                             "Previous T2 amplitudes.", occ_per_irrep_,
                             occ_per_irrep_, unocc_per_irrep_,
                             unocc_per_irrep_),
                         T2_temp = TiledTensor<double, 4>(
                             "T2 amplitude temporary tensor.", occ_per_irrep_,
                             occ_per_irrep_, unocc_per_irrep_,
                             unocc_per_irrep_);
  TiledTensor<double, 4> W_temp = TiledTensor<double, 4>(
      "Temporary for computing W", occ_per_irrep_, occ_per_irrep_,
      unocc_per_irrep_, unocc_per_irrep_);
  TiledTensor<double, 1> occ_evals("Occupied eigenvalues", occ_per_irrep_);
  TiledTensor<double, 1> unocc_evals("Unoccupied eigenvalues",
                                     unocc_per_irrep_);

  timer_off("EinHF: Set up CCSD amplitudes.");

  timer_on("EinHF: Compute CCSD amplitudes.");

  // Set up the "Fock matrix".
  for (int G = 0; G < nirrep_; G++) {
    if (occ_per_irrep_[G] != 0) {
      occ_evals.tile(G) = getEvals()(
          Range{irrep_offsets_[G], irrep_offsets_[G] + occ_per_irrep_[G]});
    }
    if (unocc_per_irrep_[G] != 0) {
      unocc_evals.tile(G) =
          getEvals()(Range{irrep_offsets_[G] + occ_per_irrep_[G],
                           irrep_offsets_[G] + irrep_sizes_[G]});
    }
  }

  double e_0 = 0.0, e_1 = 1.0;
  double dRMS = 1.0;
  int cycle = 0;

  while (std::abs(e_0 - e_1) > e_convergence_ && dRMS > d_convergence_ &&
         cycle < maxiter_) {

    e_1 = e_0;
    T1_prev = t1_amps_;
    T2_prev = t2_amps_;
    cycle++;

    // Calculate the tau intermediates.
    tau = t2_amps_;
    tau_tilde = t2_amps;

    einsum(1.0, Indices{index::i, index::j, index::a, index::b}, &tau, 1.0,
           Indices{index::i, index::a}, t1_amps, Indices{index::j, index::b},
           t1_amps);
    einsum(1.0, Indices{index::i, index::j, index::a, index::b}, &tau, -1.0,
           Indices{index::i, index::b}, t1_amps, Indices{index::j, index::a},
           t1_amps);
    einsum(1.0, Indices{index::i, index::j, index::a, index::b}, &tau_tilde,
           0.5, Indices{index::i, index::a}, t1_amps,
           Indices{index::j, index::b}, t1_amps);
    einsum(1.0, Indices{index::i, index::j, index::a, index::b}, &tau_tilde,
           -0.5, Indices{index::i, index::b}, t1_amps,
           Indices{index::j, index::a}, t1_amps);

    // Calculate the F intermediate.

    // The first term is zero since the Fock matrix is diagonal in an RHF
    // reference.

    // The second term is also zero.

    // The third term finally exists.
    einsum(0.0, Indices{index::a, index::e}, &Fae, 1.0,
           Indices{index::m, index::f}, t1_amps_,
           Indices{index::m, index::a, index::f, index::e}, tei_anti_ovvv);

    // The next term also exists.
    einsum(1.0, Indices{index::a, index::e}, &Fae, -0.5,
           Indices{index::m, index::n, index::a, index::f}, tau_tilde,
           Indices{index::m, index::n, index::e, index::f}, tei_anti_oovv);

    // The first term is zero since the Fock matrix is diagonal.
    // The second term is zero for the same reason.

    // The third term is not.
    einsum(0.0, Indices{index::m, index::i}, &Fmi, 1.0,
           Indices{index::n, index::e}, t1_amps_,
           Indices{index::m, index::n, index::i, index::e}, tei_anti_ooov);

    // The last term is not.
    einsum(1.0, Indices{index::m, index::i}, &Fmi, 0.5,
           Indices{index::i, index::n, index::e, index::f}, tau_tilde,
           Indices{index::m, index::n, index::e, index::f}, tei_anti_oovv);

    // The first term is zero.
    // The last term is not.
    einsum(0.0, Indices{index::m, index::e}, &Fme, 1.0,
           Indices{index::n, index::f}, t1_amps_,
           Indices{index::m, index::n, index::e, index::f}, tei_anti_oovv);

    // Calculate the W intermediate.

    // All terms are non-zero.
    Wmnij = tei_anti_oooo;

    einsum(1.0, Indices{index::m, index::n, index::i, index::j}, &Wmnij, 1.0,
           Indices{index::j, index::e}, t1_amps_,
           Indices{index::m, index::n, index::i, index::e}, tei_anti_ooov);
    einsum(1.0, Indices{index::m, index::n, index::i, index::j}, &Wmnij, -1.0,
           Indices{index::i, index::e}, t1_amps_,
           Indices{index::m, index::n, index::j, index::e}, tei_anti_ooov);

    einsum(1.0, Indices{index::m, index::n, index::i, index::j}, &Wmnij, 0.25,
           Indices{index::i, index::j, index::e, index::f}, tau,
           Indices{index::m, index::n, index::e, index::f}, tei_anti_oovv);

    Wabef = tei_anti_vvvv;

    einsum(1.0, Indices{index::a, index::b, index::e, index::f}, &Wabef, 1.0,
           Indices{index::m, index::b}, t1_amps_,
           Indices{index::m, index::a, index::e, index::f}, tei_anti_ovvv);
    einsum(1.0, Indices{index::a, index::b, index::e, index::f}, &Wabef, -1.0,
           Indices{index::m, index::a}, t1_amps_,
           Indices{index::m, index::b, index::e, index::f}, tei_anti_ovvv);
    einsum(1.0, Indices{index::a, index::b, index::e, index::f}, &Wabef, 0.25,
           Indices{index::m, index::n, index::a, index::b}, tau,
           Indices{index::m, index::n, index::e, index::f}, tei_anti_oovv);

    sort(0.0, Indices{index::m, index::b, index::e, index::j}, &Wmbej, -1.0,
         Indices{index::m, index::b, index::j, index::e}, tei_anti_ovov);
    einsum(1.0, Indices{index::m, index::b, index::e, index::j}, &Wmbej, 1.0,
           Indices{index::j, index::f}, t1_amps_,
           Indices{index::m, index::b, index::e, index::f}, tei_anti_ovvv);
    einsum(1.0, Indices{index::m, index::b, index::e, index::j}, &Wmbej, 1.0,
           Indices{index::n, index::b}, t1_amps_,
           Indices{index::m, index::n, index::j, index::e}, tei_anti_ooov);
    W_temp = t2_amps_;
    einsum(0.5, Indices{index::j, index::n, index::f, index::b}, &W_temp, 1.0,
           Indices{index::j, index::f}, t1_amps_, Indices{index::n, index::b},
           t1_amps_);
    einsum(1.0, Indices{index::m, index::b, index::e, index::j}, &Wmbej, -1.0,
           Indices{index::j, index::n, index::f, index::b}, W_temp,
           Indices{index::m, index::n, index::e, index::f}, tei_anti_oovv);

    // Calculate the T1 amplitudes.
    // First term is zero.
    // Second term is not.
    einsum(0.0, Indices{index::i, index::a}, &T1_temp, 1.0,
           Indices{index::i, index::e}, T1_prev, Indices{index::a, index::e},
           Fae);
    // Third term is not.
    einsum(1.0, Indices{index::i, index::a}, &T1_temp, -1.0,
           Indices{index::m, index::a}, T1_prev, Indices{index::m, index::i},
           Fmi);
    // And so on.
    einsum(1.0, Indices{index::i, index::a}, &T1_temp, 1.0,
           Indices{index::i, index::m, index::a, index::e}, T2_prev,
           Indices{index::m, index::e}, Fme);
    einsum(1.0, Indices{index::i, index::a}, &T1_temp, -1.0,
           Indices{index::n, index::f}, T1_prev,
           Indices{index::n, index::a, index::i, index::f}, tei_anti_ovov);
    einsum(1.0, Indices{index::i, index::a}, &T1_temp, -0.5,
           Indices{index::i, index::m, index::e, index::f}, T2_prev,
           Indices{index::m, index::a, index::e, index::f}, tei_anti_ovvv);
    einsum(1.0, Indices{index::i, index::a}, &T1_temp, 0.5,
           Indices{index::m, index::n, index::a, index::e}, T2_prev,
           Indices{index::m, index::n, index::i, index::e}, tei_anti_ooov);
    einsum(0.0, Indices{index::i, index::a}, &t1_amps_, 1.0,
           Indices{index::i, index::a}, T1_temp, Indices{index::i, index::a},
           t1_den_);
  }
  timer_off("EinHF: Compute CCSD amplitudes.");

  timer_off("EinHF: Computing CCSD energy");

  outfile->Printf("\tMP2 Same-spin:\t%lf\n", (double)eMP2_SS);
  outfile->Printf("\tMP2 Opposite-spin:\t%lf\n", (double)eMP2_OS);
  outfile->Printf("\tMP2 Correction:\t%lf\n", e_new);

  energy_ += e_new;
  outfile->Printf("\tTotal MP2 Energy:\t%lf\n", energy_);

  return energy_;
}

void EinsumsRCCSD::print_header() {
  int nthread = Process::environment.get_n_threads();

  outfile->Printf("\n");
  outfile->Printf(
      "         ---------------------------------------------------------\n");
  outfile->Printf("                                   Einsums RCCSD\n");
  outfile->Printf("                                by Connor Briggs\n");
  outfile->Printf("                                 %4s Reference\n",
                  options_.get_str("REFERENCE").c_str());
  outfile->Printf("                               Running on the CPU\n");
  outfile->Printf("                      %3d Threads, %6ld MiB Core\n", nthread,
                  memory_ / 1048576L);
  outfile->Printf(
      "         ---------------------------------------------------------\n");
  outfile->Printf("\n");
}
} // namespace einhf
} // namespace psi
