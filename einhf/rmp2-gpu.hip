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

#include "rmp2.h"
#include "rmp2-gpu.h"
#include "rhf.h"

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

GPUEinsumsRMP2::GPUEinsumsRMP2(std::shared_ptr<GPUEinsumsRHF> ref_wfn, Options &options)
    : Wavefunction(options), S_(ref_wfn->getS()), F_(ref_wfn->getF()), Ft_(ref_wfn->getFt()), C_(ref_wfn->getC()), Cocc_(ref_wfn->getCocc()), D_(ref_wfn->getD()) {

  timer_on("EinHF: Setup MP2 wavefunction");

  // Shallow copy useful objects from the passed in wavefunction
  shallow_copy(ref_wfn);

  energy_ = ref_wfn->energy();

  evals_ = ref_wfn->getEvals();
  evals_.set_name("Orbital energies");

  print_ = options_.get_int("PRINT");

  nirrep_ = sobasisset_->nirrep();
  nso_ = basisset_->nbf();
  occ_per_irrep_ = ref_wfn->getOccPerIrrep();
  irrep_sizes_ = ref_wfn->getIrrepSizes();
  unocc_per_irrep_ = std::vector<int>(nirrep_);
  irrep_offsets_ = std::vector<int>(nirrep_);
  irrep_offsets_[0] = 0;
  ndocc_ = ref_wfn->getNDocc();

  for (int i = 0; i < nirrep_; i++) {
    unocc_per_irrep_[i] = irrep_sizes_.at(i) - occ_per_irrep_.at(i);
    if (i != 0) {
      irrep_offsets_[i] = irrep_offsets_[i - 1] + irrep_sizes_.at(i - 1);
    }
  }

  print_header();

  for (int i = 0; i < S_.num_blocks(); i++) {
    irrep_names_.push_back(ref_wfn->getS()[i].name());
    S_[i].set_name(ref_wfn->getS()[i].name());
    F_[i].set_name(ref_wfn->getS()[i].name());
    Ft_[i].set_name(ref_wfn->getS()[i].name());
    C_[i].set_name(ref_wfn->getS()[i].name());
    Cocc_[i].set_name(ref_wfn->getS()[i].name());
    D_[i].set_name(ref_wfn->getS()[i].name());
  }

  S_.set_name("Overlap");
  F_.set_name("Fock matrix");
  Ft_.set_name("Transformed Fock matrix");
  C_.set_name("MO coefficients");
  Cocc_.set_name("Occupied MO coefficients");
  D_.set_name("Density matrix");

  outfile->Printf("    Number of orbitals: %d\n", nso_);
  outfile->Printf("    Orbital Occupation:\n         \t");

  for (int i = 0; i < S_.num_blocks(); i++) {
    outfile->Printf("%4s\t", S_.name(i).c_str());
  }

  outfile->Printf("\n    DOCC \t");
  for (int i = 0; i < occ_per_irrep_.size(); i++) {
    outfile->Printf("%4d\t", occ_per_irrep_[i]);
  }

  outfile->Printf("\n    VIRT \t");

  for (int i = 0; i < occ_per_irrep_.size(); i++) {
    outfile->Printf("%4d\t", irrep_sizes_[i] - occ_per_irrep_[i]);
  }

  outfile->Printf("\n    Total\t");

  for (int i = 0; i < occ_per_irrep_.size(); i++) {
    outfile->Printf("%4d\t", irrep_sizes_[i]);
  }

  outfile->Printf("\n");

  init_integrals();

  timer_off("EinHF: Setup MP2 wavefunction");
}

GPUEinsumsRMP2::~GPUEinsumsRMP2() {}

static void calculate_tei(std::shared_ptr<TwoBodySOInt> ints,
                          einsums::TiledTensor<double, 4> *out) {
  auto functor = [out](int iiabs, int jjabs, int kkabs, int llabs, int iiirrep,
                       int iirel, int jjirrep, int jjrel, int kkirrep,
                       int kkrel, int llirrep, int llrel, double val) {
    (*out)(iiabs, jjabs, kkabs, llabs) = val;
    (*out)(iiabs, jjabs, llabs, kkabs) = val;
    (*out)(jjabs, iiabs, kkabs, llabs) = val;
    (*out)(jjabs, iiabs, llabs, kkabs) = val;
    (*out)(kkabs, llabs, iiabs, jjabs) = val;
    (*out)(kkabs, llabs, jjabs, iiabs) = val;
    (*out)(llabs, kkabs, iiabs, jjabs) = val;
    (*out)(llabs, kkabs, jjabs, iiabs) = val;
  };
  ints->compute_integrals(functor);
}

void GPUEinsumsRMP2::set_tile(const TiledDeviceTensor<double, 4> &temp, int i, int a,
                           int j, int b) {
  if (occ_per_irrep_[i] != 0 && occ_per_irrep_[a] != irrep_sizes_[a] &&
      occ_per_irrep_[j] != 0 && occ_per_irrep_[b] != irrep_sizes_[b] &&
      temp.has_tile(i, a, j, b)) {
    std::string tile_name = "(" + S_[i].name() + ", " + S_[a].name() + ", " +
                            S_[j].name() + ", " + S_[b].name() + ")";

    teit_.tile(i, a, j, b) = temp.tile(i, a, j, b)(
        Range{0, occ_per_irrep_[i]}, Range{occ_per_irrep_[a], irrep_sizes_[a]},
        Range{0, occ_per_irrep_[j]}, Range{occ_per_irrep_[b], irrep_sizes_[b]});
    denominator_.tile(i, a, j, b);

    teit_.tile(i, a, j, b).set_name(tile_name);
    denominator_.tile(i, a, j, b).set_name(tile_name);
  }
}

void GPUEinsumsRMP2::init_integrals() {
  // The basisset object contains all of the basis information and is formed in
  // the new_wavefunction call The integral factory oversees the creation of
  // integral objects
  auto integral = std::make_shared<IntegralFactory>(basisset_, basisset_,
                                                    basisset_, basisset_);

  // Determine the number of electrons in the system
  // The molecule object is built into all wavefunctions
  int charge = molecule_->molecular_charge();
  int nelec = 0;
  for (int i = 0; i < molecule_->natom(); ++i) {
    nelec += (int)molecule_->Z(i);
  }
  nelec -= charge;
  if (nelec % 2) {
    throw PSIEXCEPTION("This is only an RMP2 code, but you gave it an odd "
                       "number of electrons.  Try again!");
  }
  // ndocc_ = nelec / 2;

  tei_ = einsums::TiledDeviceTensor<double, 4>("TEI", irrep_sizes_);
  teit_ =
      einsums::TiledDeviceTensor<double, 4>("TEI", occ_per_irrep_, unocc_per_irrep_,
                                      occ_per_irrep_, unocc_per_irrep_);
  MP2_amps_ = einsums::TiledDeviceTensor<double, 4>("MP2 Amps", occ_per_irrep_,
                                              unocc_per_irrep_, occ_per_irrep_,
                                              unocc_per_irrep_);

  auto ints = std::make_shared<IntegralFactory>(basisset_);

  timer_on("EinHF: Generating Integrals");

  std::vector<std::shared_ptr<TwoBodyAOInt>> computer =
      std::vector<std::shared_ptr<TwoBodyAOInt>>(omp_get_max_threads());

  for (int i = 0; i < computer.size(); i++) {
    computer[i] = ints->eri();
  }

  auto symm_computer = std::make_shared<TwoBodySOInt>(computer, ints);

  auto tei_temp = einsums::TiledTensor<double, 4>("TEI", irrep_sizes_);

  calculate_tei(symm_computer, &tei_temp);

  tei_ = tei_temp;

  timer_off("EinHF: Generating Integrals");

  timer_on("EinHF: Transforming Two-electron Integrals");

  TiledDeviceTensor<double, 4> temp1("Transform temp1", irrep_sizes_),
      temp2("Transform temp2", irrep_sizes_);

  einsum(Indices{index::p, index::q, index::r, index::s}, &temp1,
         Indices{index::m, index::q, index::r, index::s}, tei_,
         Indices{index::m, index::p}, C_);
  einsum(Indices{index::p, index::q, index::r, index::s}, &temp2,
         Indices{index::p, index::m, index::r, index::s}, temp1,
         Indices{index::m, index::q}, C_);
  einsum(Indices{index::p, index::q, index::r, index::s}, &temp1,
         Indices{index::p, index::q, index::m, index::s}, temp2,
         Indices{index::m, index::r}, C_);
  einsum(Indices{index::p, index::q, index::r, index::s}, &temp2,
         Indices{index::p, index::q, index::r, index::m}, temp1,
         Indices{index::m, index::s}, C_);

  denominator_ =
      RMP2ScaleTensor("MP2 denominator", occ_per_irrep_, unocc_per_irrep_,
                      irrep_offsets_, irrep_sizes_, irrep_names_, &evals_);

  for (int i = 0; i < nirrep_; i++) {
    for(int a = 0; a < nirrep_; a++) {
      for(int j = 0; j < nirrep_; j++) {
        for(int b = 0; b < nirrep_; b++) {
          set_tile(temp2, i, a, j, b);
        }
      }
    }
  }

  TiledTensor<double, 4> denom_host_temp{"Denom temp", occ_per_irrep_, unocc_per_irrep_, occ_per_irrep_, unocc_per_irrep_};

  denom_host_temp = denominator_;

  TiledDeviceTensorView<double, 4> denom_temp = denom_host_temp;

  einsum(Indices{index::i, index::a, index::j, index::b}, &MP2_amps_,
         Indices{index::i, index::a, index::j, index::b}, teit_,
         Indices{index::i, index::a, index::j, index::b}, denom_temp);

  timer_off("EinHF: Transforming Two-electron Integrals");
}

double GPUEinsumsRMP2::compute_energy() {
  timer_on("EinHF: Computing MP2 energy");

  double e_new;

  DeviceTensor<double, 0> eMP2_SS, eMP2_OS;

  einsum(0.0, Indices{}, &eMP2_OS, 1.0,
         Indices{index::i, index::a, index::j, index::b}, MP2_amps_,
         Indices{index::i, index::a, index::j, index::b}, teit_);

  einsum(0.0, Indices{}, &eMP2_SS, -1.0,
         Indices{index::i, index::a, index::j, index::b}, MP2_amps_,
         Indices{index::i, index::b, index::j, index::a}, teit_);

  eMP2_SS += eMP2_OS;

  e_new = (double)eMP2_SS + (double)eMP2_OS;

  timer_off("EinHF: Computing MP2 energy");

  outfile->Printf("\tMP2 Same-spin:\t%lf\n", (double)eMP2_SS);
  outfile->Printf("\tMP2 Opposite-spin:\t%lf\n", (double)eMP2_OS);
  outfile->Printf("\tMP2 Correction:\t%lf\n", e_new);

  energy_ += e_new;
  outfile->Printf("\tTotal MP2 Energy:\t%lf\n", energy_);

  return energy_;
}

void GPUEinsumsRMP2::print_header() {
  int nthread = Process::environment.get_n_threads();

  outfile->Printf("\n");
  outfile->Printf(
      "         ---------------------------------------------------------\n");
  outfile->Printf("                                   Einsums RMP2\n");
  outfile->Printf("                                by Connor Briggs\n");
  outfile->Printf("                                 %4s Reference\n",
                  options_.get_str("REFERENCE").c_str());
  outfile->Printf("                               Running on the GPU\n");
  outfile->Printf("                      %3d Threads, %6ld MiB Core\n", nthread,
                  memory_ / 1048576L);
  outfile->Printf(
      "         ---------------------------------------------------------\n");
  outfile->Printf("\n");
}
} // namespace einhf
} // namespace psi
