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

#include "ump2.h"
#include "uhf.h"

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

EinsumsUMP2::EinsumsUMP2(std::shared_ptr<EinsumsUHF> ref_wfn, Options &options)
    : Wavefunction(options) {

  timer_on("EinHF: Setup UMP2 wavefunction");

  // Shallow copy useful objects from the passed in wavefunction
  shallow_copy(ref_wfn);

  energy_ = ref_wfn->energy();

  evalsa_ = ref_wfn->getEvalsa();
  evalsa_.set_name("Alpha Orbital energies");
  evalsb_ = ref_wfn->getEvalsb();
  evalsb_.set_name("Beta Orbital energies");

  print_ = options_.get_int("PRINT");

  nirrep_ = sobasisset_->nirrep();
  nso_ = basisset_->nbf();
  aocc_per_irrep_ = ref_wfn->getAOccPerIrrep();
  bocc_per_irrep_ = ref_wfn->getBOccPerIrrep();
  irrep_sizes_ = ref_wfn->getIrrepSizes();
  aunocc_per_irrep_ = std::vector<int>(nirrep_);
  bunocc_per_irrep_ = std::vector<int>(nirrep_);
  irrep_offsets_ = std::vector<int>(nirrep_);
  irrep_offsets_[0] = 0;

  naocc_ = ref_wfn->getNAocc();
  nbocc_ = ref_wfn->getNBocc();

  for (int i = 0; i < nirrep_; i++) {
    aunocc_per_irrep_[i] = irrep_sizes_.at(i) - aocc_per_irrep_.at(i);
    bunocc_per_irrep_[i] = irrep_sizes_.at(i) - bocc_per_irrep_.at(i);
    if (i != 0) {
      irrep_offsets_[i] = irrep_offsets_[i - 1] + irrep_sizes_.at(i - 1);
    }
  }

  print_header();

  S_ = ref_wfn->getS();
  Fa_ = ref_wfn->getFa();
  Fta_ = ref_wfn->getFta();
  Fb_ = ref_wfn->getFb();
  Ftb_ = ref_wfn->getFtb();
  Ca_ = ref_wfn->getCa();
  Cocca_ = ref_wfn->getCocca();
  Da_ = ref_wfn->getDa();
  Cb_ = ref_wfn->getCb();
  Coccb_ = ref_wfn->getCoccb();
  Db_ = ref_wfn->getDb();

  for (int i = 0; i < S_.num_blocks(); i++) {
    irrep_names_.push_back(ref_wfn->getS()[i].name());
    S_[i].set_name(ref_wfn->getS()[i].name());
    Fa_[i].set_name(ref_wfn->getS()[i].name());
    Fta_[i].set_name(ref_wfn->getS()[i].name());
    Ca_[i].set_name(ref_wfn->getS()[i].name());
    Cocca_[i].set_name(ref_wfn->getS()[i].name());
    Da_[i].set_name(ref_wfn->getS()[i].name());
    Fb_[i].set_name(ref_wfn->getS()[i].name());
    Ftb_[i].set_name(ref_wfn->getS()[i].name());
    Cb_[i].set_name(ref_wfn->getS()[i].name());
    Coccb_[i].set_name(ref_wfn->getS()[i].name());
    Db_[i].set_name(ref_wfn->getS()[i].name());
  }

  S_.set_name("Overlap");
  Fa_.set_name("Alpha Fock matrix");
  Fta_.set_name("Alpha Transformed Fock matrix");
  Ca_.set_name("Alpha MO coefficients");
  Cocca_.set_name("AlphaOccupied MO coefficients");
  Da_.set_name("Alpha Density matrix");
  Fb_.set_name("Beta Fock matrix");
  Ftb_.set_name("Beta Transformed Fock matrix");
  Cb_.set_name("Beta MO coefficients");
  Coccb_.set_name("Beta Occupied MO coefficients");
  Db_.set_name("Beta Density matrix");

  outfile->Printf("    Number of orbitals: %d\n", nso_);
  outfile->Printf("    Orbital Occupation:\n         \t");

  for (int i = 0; i < S_.num_blocks(); i++) {
    outfile->Printf("%4s\t", S_.name(i).c_str());
  }

  outfile->Printf("\n    DOCC \t");
  for (int i = 0; i < bocc_per_irrep_.size(); i++) {
    outfile->Printf("%4d\t", bocc_per_irrep_[i]);
  }

  outfile->Printf("\n    SOCC \t");

  for (int i = 0; i < aocc_per_irrep_.size(); i++) {
    outfile->Printf("%4d\t", aocc_per_irrep_[i] - bocc_per_irrep_[i]);
  }

  outfile->Printf("\n    Virt \t");
  for (int i = 0; i < bocc_per_irrep_.size(); i++) {
    outfile->Printf("%4d\t", irrep_sizes_[i] - aocc_per_irrep_[i]);
  }

  outfile->Printf("\n    Total\t");

  for (int i = 0; i < aocc_per_irrep_.size(); i++) {
    outfile->Printf("%4d\t", irrep_sizes_[i]);
  }

  outfile->Printf("\n");

  init_integrals();

  timer_off("EinHF: Setup UMP2 wavefunction");
}

EinsumsUMP2::~EinsumsUMP2() {}

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

void EinsumsUMP2::set_tile(const TiledTensor<double, 4> &temp,
                           TiledTensor<double, 4> &teit,
                           UMP2ScaleTensor &denominator,
                           const std::vector<int> &aocc_per_irrep,
                           const std::vector<int> &bocc_per_irrep, int i, int a,
                           int j, int b) {
  if (aocc_per_irrep[i] != 0 && aocc_per_irrep[a] != irrep_sizes_[a] &&
      bocc_per_irrep[j] != 0 && bocc_per_irrep[b] != irrep_sizes_[b] && temp.has_tile(i, a, j, b)) {
    std::string tile_name = "(" + S_[i].name() + ", " + S_[a].name() + ", " +
                            S_[j].name() + ", " + S_[b].name() + ")";

    teit.tile(i, a, j, b) = temp.tile(i, a, j, b)(
        Range{0, aocc_per_irrep[i]}, Range{aocc_per_irrep[a], irrep_sizes_[a]},
        Range{0, bocc_per_irrep[j]}, Range{bocc_per_irrep[b], irrep_sizes_[b]});
    denominator.tile(i, a, j, b);

    teit.tile(i, a, j, b).set_name(tile_name);
    denominator.tile(i, a, j, b).set_name(tile_name);
  }
}

void EinsumsUMP2::init_integrals() {
  // The basisset object contains all of the basis information and is formed in
  // the new_wavefunction call The integral factory oversees the creation of
  // integral objects
  auto integral = std::make_shared<IntegralFactory>(basisset_, basisset_,
                                                    basisset_, basisset_);

  tei_ = einsums::TiledTensor<double, 4>("TEI", irrep_sizes_);
  teitaa_ =
      einsums::TiledTensor<double, 4>("TEI", aocc_per_irrep_, aunocc_per_irrep_,
                                      aocc_per_irrep_, aunocc_per_irrep_);
  MP2_ampsaa_ = einsums::TiledTensor<double, 4>(
      "MP2 Amps", aocc_per_irrep_, aunocc_per_irrep_, aocc_per_irrep_,
      aunocc_per_irrep_);
  teitbb_ =
      einsums::TiledTensor<double, 4>("TEI", bocc_per_irrep_, bunocc_per_irrep_,
                                      bocc_per_irrep_, bunocc_per_irrep_);
  MP2_ampsbb_ = einsums::TiledTensor<double, 4>(
      "MP2 Amps", bocc_per_irrep_, bunocc_per_irrep_, bocc_per_irrep_,
      bunocc_per_irrep_);
  teitab_ =
      einsums::TiledTensor<double, 4>("TEI", aocc_per_irrep_, aunocc_per_irrep_,
                                      bocc_per_irrep_, bunocc_per_irrep_);
  MP2_ampsab_ = einsums::TiledTensor<double, 4>(
      "MP2 Amps", aocc_per_irrep_, aunocc_per_irrep_, bocc_per_irrep_,
      bunocc_per_irrep_);

  auto ints = std::make_shared<IntegralFactory>(basisset_);

  timer_on("EinHF: Generating Integrals");

  std::vector<std::shared_ptr<TwoBodyAOInt>> computer =
      std::vector<std::shared_ptr<TwoBodyAOInt>>(omp_get_max_threads());

  for (int i = 0; i < computer.size(); i++) {
    computer[i] = ints->eri();
  }

  auto symm_computer = std::make_shared<TwoBodySOInt>(computer, ints);

  calculate_tei(symm_computer, &tei_);

  timer_off("EinHF: Generating Integrals");

  timer_on("EinHF: Transforming Two-electron Integrals");

#pragma omp taskgroup
  {
#pragma omp task depend(in : this->tei_, this->evalsa_)                        \
    depend(out : this -> denominatoraa_, this->teitaa_, this->MP2_ampsaa_)
    {
      TiledTensor<double, 4> temp1("Transform temp1", irrep_sizes_),
          temp2("Transform temp2", irrep_sizes_);

      einsum(Indices{index::p, index::q, index::r, index::s}, &temp1,
             Indices{index::m, index::q, index::r, index::s}, tei_,
             Indices{index::m, index::p}, Ca_);
      einsum(Indices{index::p, index::q, index::r, index::s}, &temp2,
             Indices{index::p, index::m, index::r, index::s}, temp1,
             Indices{index::m, index::q}, Ca_);
      einsum(Indices{index::p, index::q, index::r, index::s}, &temp1,
             Indices{index::p, index::q, index::m, index::s}, temp2,
             Indices{index::m, index::r}, Ca_);
      einsum(Indices{index::p, index::q, index::r, index::s}, &temp2,
             Indices{index::p, index::q, index::r, index::m}, temp1,
             Indices{index::m, index::s}, Ca_);
      denominatoraa_ = UMP2ScaleTensor(
          "AA MP2 denominator", aocc_per_irrep_, aunocc_per_irrep_,
          aocc_per_irrep_, aunocc_per_irrep_, irrep_offsets_, irrep_sizes_,
          irrep_names_, &evalsa_, &evalsa_);

      for (int i = 0; i < nirrep_; i++) {
        for (int a = 0; a < nirrep_; a++) {
          for (int j = 0; j < nirrep_; j++) {
            for (int b = 0; b < nirrep_; b++) {
              set_tile(temp2, teitaa_, denominatoraa_, aocc_per_irrep_,
                       aocc_per_irrep_, i, a, j, b);
            }
          }
        }
      }

      einsum(Indices{index::i, index::a, index::j, index::b}, &MP2_ampsaa_,
             Indices{index::i, index::a, index::j, index::b}, teitaa_,
             Indices{index::i, index::a, index::j, index::b}, denominatoraa_);
    }
#pragma omp task depend(in : this->tei_, this->evalsb_)                        \
    depend(out : this -> denominatorbb_, this->teitbb_, this->MP2_ampsbb_)
    {
      TiledTensor<double, 4> temp1("Transform temp1", irrep_sizes_),
          temp2("Transform temp2", irrep_sizes_);

      einsum(Indices{index::p, index::q, index::r, index::s}, &temp1,
             Indices{index::m, index::q, index::r, index::s}, tei_,
             Indices{index::m, index::p}, Cb_);
      einsum(Indices{index::p, index::q, index::r, index::s}, &temp2,
             Indices{index::p, index::m, index::r, index::s}, temp1,
             Indices{index::m, index::q}, Cb_);
      einsum(Indices{index::p, index::q, index::r, index::s}, &temp1,
             Indices{index::p, index::q, index::m, index::s}, temp2,
             Indices{index::m, index::r}, Cb_);
      einsum(Indices{index::p, index::q, index::r, index::s}, &temp2,
             Indices{index::p, index::q, index::r, index::m}, temp1,
             Indices{index::m, index::s}, Cb_);
      denominatorbb_ = UMP2ScaleTensor(
          "BB MP2 denominator", bocc_per_irrep_, bunocc_per_irrep_,
          bocc_per_irrep_, bunocc_per_irrep_, irrep_offsets_, irrep_sizes_,
          irrep_names_, &evalsb_, &evalsb_);

      for (int i = 0; i < nirrep_; i++) {
        for (int a = 0; a < nirrep_; a++) {
          for (int j = 0; j < nirrep_; j++) {
            for (int b = 0; b < nirrep_; b++) {
              set_tile(temp2, teitbb_, denominatorbb_, bocc_per_irrep_,
                       bocc_per_irrep_, i, a, j, b);
            }
          }
        }
      }

      einsum(Indices{index::i, index::a, index::j, index::b}, &MP2_ampsbb_,
             Indices{index::i, index::a, index::j, index::b}, teitbb_,
             Indices{index::i, index::a, index::j, index::b}, denominatorbb_);
    }
#pragma omp task depend(in : this->tei_, this->evalsa_, this->evalsb_)         \
    depend(out : this -> denominatorab_, this->teitab_, this->MP2_ampsab_)
    {
      TiledTensor<double, 4> temp1("Transform temp1", irrep_sizes_),
          temp2("Transform temp2", irrep_sizes_);

      einsum(Indices{index::p, index::q, index::r, index::s}, &temp1,
             Indices{index::m, index::q, index::r, index::s}, tei_,
             Indices{index::m, index::p}, Ca_);
      einsum(Indices{index::p, index::q, index::r, index::s}, &temp2,
             Indices{index::p, index::m, index::r, index::s}, temp1,
             Indices{index::m, index::q}, Ca_);
      einsum(Indices{index::p, index::q, index::r, index::s}, &temp1,
             Indices{index::p, index::q, index::m, index::s}, temp2,
             Indices{index::m, index::r}, Cb_);
      einsum(Indices{index::p, index::q, index::r, index::s}, &temp2,
             Indices{index::p, index::q, index::r, index::m}, temp1,
             Indices{index::m, index::s}, Cb_);
      denominatorab_ = UMP2ScaleTensor(
          "AB MP2 denominator", aocc_per_irrep_, aunocc_per_irrep_,
          bocc_per_irrep_, bunocc_per_irrep_, irrep_offsets_, irrep_sizes_,
          irrep_names_, &evalsa_, &evalsb_);

      for (int i = 0; i < nirrep_; i++) {
        for (int a = 0; a < nirrep_; a++) {
          for (int j = 0; j < nirrep_; j++) {
            for (int b = 0; b < nirrep_; b++) {
              set_tile(temp2, teitab_, denominatorab_, aocc_per_irrep_,
                       bocc_per_irrep_, i, a, j, b);
            }
          }
        }
      }

      einsum(Indices{index::i, index::a, index::j, index::b}, &MP2_ampsab_,
             Indices{index::i, index::a, index::j, index::b}, teitab_,
             Indices{index::i, index::a, index::j, index::b}, denominatorab_);
    }
  }

  timer_off("EinHF: Transforming Two-electron Integrals");
}

double EinsumsUMP2::compute_energy() {
  timer_on("EinHF: Computing UMP2 energy");

  double e_new;

  Tensor<double, 0> eMP2_AS, eMP2_BS, eMP2_OS;

  einsum(0.0, Indices{}, &eMP2_OS, 1.0,
         Indices{index::i, index::a, index::j, index::b}, MP2_ampsab_,
         Indices{index::i, index::a, index::j, index::b}, teitab_);

  einsum(0.0, Indices{}, &eMP2_AS, 0.5,
         Indices{index::i, index::a, index::j, index::b}, MP2_ampsaa_,
         Indices{index::i, index::a, index::j, index::b}, teitaa_);
  einsum(1.0, Indices{}, &eMP2_AS, -0.5,
         Indices{index::i, index::a, index::j, index::b}, MP2_ampsaa_,
         Indices{index::i, index::b, index::j, index::a}, teitaa_);

  einsum(0.0, Indices{}, &eMP2_BS, 0.5,
         Indices{index::i, index::a, index::j, index::b}, MP2_ampsbb_,
         Indices{index::i, index::a, index::j, index::b}, teitbb_);
  einsum(1.0, Indices{}, &eMP2_BS, -0.5,
         Indices{index::i, index::a, index::j, index::b}, MP2_ampsbb_,
         Indices{index::i, index::b, index::j, index::a}, teitbb_);

  e_new = (double)eMP2_AS + (double)eMP2_BS + (double)eMP2_OS;

  timer_off("EinHF: Computing UMP2 energy");

  outfile->Printf("\tMP2 Alpha-Alpha:\t%lf\n", (double)eMP2_AS);
  outfile->Printf("\tMP2 Beta-Beta:\t%lf\n", (double)eMP2_BS);
  outfile->Printf("\tMP2 Alpha-Beta:\t%lf\n", (double)eMP2_OS);
  outfile->Printf("\tMP2 Correction:\t%lf\n", e_new);

  energy_ += e_new;
  outfile->Printf("\tTotal UMP2 Energy:\t%lf\n", energy_);

  return energy_;
}

void EinsumsUMP2::print_header() {
  int nthread = Process::environment.get_n_threads();

  outfile->Printf("\n");
  outfile->Printf(
      "         ---------------------------------------------------------\n");
  outfile->Printf("                                   Einsums UMP2\n");
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