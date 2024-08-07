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

#include "uhf.h"
#include "ump2.h"

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
    : EinsumsUHF(*ref_wfn, options) {

  timer_on("EinHF: Setup MP2 wavefunction");

  // Shallow copy useful objects from the passed in wavefunction
  shallow_copy(ref_wfn);

  // energy_ = ref_wfn->energy();

  // evals_ = ref_wfn->getEvals();

  // print_ = options_.get_int("PRINT");

  // nirrep_ = sobasisset_->nirrep();
  // nso_ = basisset_->nbf();
  aunocc_per_irrep_ = std::vector<int>(nirrep_);
  irrep_offsets_ = std::vector<int>(nirrep_);
  irrep_offsets_[0] = 0;
  bunocc_per_irrep_ = std::vector<int>(nirrep_);
  irrep_offsets_ = std::vector<int>(nirrep_);

  for (int i = 0; i < nirrep_; i++) {
    aunocc_per_irrep_[i] = irrep_sizes_.at(i) - aocc_per_irrep_.at(i);
    bunocc_per_irrep_[i] = irrep_sizes_.at(i) - bocc_per_irrep_.at(i);
    if (i != 0) {
      irrep_offsets_[i] = irrep_offsets_[i - 1] + irrep_sizes_.at(i - 1);
    }
  }

  print_header();

  init_integrals();

  // H_ = ref_wfn->getH();
  // S_ = ref_wfn->getS();
  // X_ = ref_wfn->getX();
  // Fa_ = ref_wfn->getFa();
  // Fta_ = ref_wfn->getFta();
  // Ca_ = ref_wfn->getCa();
  // Cocca_ = ref_wfn->getCocca();
  // Da_ = ref_wfn->getDa();

  // Fb_ = ref_wfn->getFb();
  // Ftb_ = ref_wfn->getFtb();
  // Cb_ = ref_wfn->getCb();
  // Coccb_ = ref_wfn->getCoccb();
  // Db_ = ref_wfn->getDb();

  timer_off("EinHF: Setup MP2 wavefunction");
}

EinsumsUMP2::~EinsumsUMP2() {}

static void calculate_tei(std::shared_ptr<TwoBodySOInt> ints,
                          einsums::TiledTensor<double, 4> *out) {
  auto functor = [out](int iiabs, int jjabs, int kkabs, int llabs, int iiirrep,
                       int iirel, int jjirrep, int jjrel, int kkirrep,
                       int kkrel, int llirrep, int llrel, double val) {
    (*out)(iiabs, jjabs, kkabs, llabs) = val;
  };
  ints->compute_integrals(functor);
}

// true is alpha, false is beta.
void EinsumsUMP2::setup_spin_integrals(bool spin1, bool spin2) {
  if (!spin1 && spin2) {
    throw PSIEXCEPTION(
        "setup_spin_integrals: Only allowed combinations are true+true "
        "(alpha-alpha), true+false (alpha-beta), and false+false (beta-beta). "
        "False+true (beta-alpha) is handled through symmetry.");
  }
  // The orbital coefficient matrices.
  const BlockTensor<double, 2> &C1 = (spin1) ? Ca_ : Cb_, &C2 = (spin2) ? Ca_ : Cb_;

  TiledTensor<double, 4> temp1("Transform temp1", irrep_sizes_),
      temp2("Transform temp2", irrep_sizes_);

  TiledTensor<double, 4> &teit =
                             (spin1) ? ((spin2) ? teitaa_ : teitab_) : teitbb_,
                         &MP2_amps = (spin1)
                                         ? ((spin2) ? MP2_ampsaa_ : MP2_ampsab_)
                                         : MP2_ampsbb_,
                         &denominator = (spin1) ? ((spin2) ? denominatoraa_
                                                           : denominatorab_)
                                                : denominatorbb_;
  // Sizes.
  const std::vector<int> &occ_per_irrep1 =
                             (spin1) ? aocc_per_irrep_ : bocc_per_irrep_,
                         &occ_per_irrep2 =
                             (spin2) ? aocc_per_irrep_ : bocc_per_irrep_,
                         &unocc_per_irrep1 =
                             (spin1) ? aunocc_per_irrep_ : bunocc_per_irrep_,
                         &unocc_per_irrep2 =
                             (spin2) ? aunocc_per_irrep_ : bunocc_per_irrep_;

  einsum(Indices{index::p, index::q, index::r, index::s}, &temp1,
         Indices{index::m, index::q, index::r, index::s}, tei_,
         Indices{index::m, index::p}, C1);
  einsum(Indices{index::p, index::q, index::r, index::s}, &temp2,
         Indices{index::p, index::m, index::r, index::s}, temp1,
         Indices{index::m, index::q}, C1);
  einsum(Indices{index::p, index::q, index::r, index::s}, &temp1,
         Indices{index::p, index::q, index::m, index::s}, temp2,
         Indices{index::m, index::r}, C2);
  einsum(Indices{index::p, index::q, index::r, index::s}, &temp2,
         Indices{index::p, index::q, index::r, index::m}, temp1,
         Indices{index::m, index::s}, C2);

  UMP2ScaleFunction full_denom("MP2 denominator",
                               &((spin1) ? evalsa_ : evalsb_),
                               &((spin2) ? evalsa_ : evalsb_));

  for (int i = 0; i < nirrep_; i++) {
    for (int j = i; j < nirrep_; j++) {
      if (i == j && occ_per_irrep1[i] != 0 && irrep_sizes_[i] != 0 &&
          occ_per_irrep1[i] != irrep_sizes_[i] && occ_per_irrep2[i] != 0 &&
          occ_per_irrep2[i] != irrep_sizes_[i]) {
        teit.tile(i, i, i, i) =
            temp2.tile(i, i, i, i)(Range{0, occ_per_irrep1[i]},
                                   Range{occ_per_irrep1[i], irrep_sizes_[i]},
                                   Range{0, occ_per_irrep2[i]},
                                   Range{occ_per_irrep2[i], irrep_sizes_[i]});
        denominator.tile(i, i, i, i) = full_denom(
            Range{irrep_offsets_[i], irrep_offsets_[i] + occ_per_irrep1[i]},
            Range{irrep_offsets_[i] + occ_per_irrep1[i],
                  irrep_offsets_[i] + irrep_sizes_[i]},
            Range{irrep_offsets_[i], irrep_offsets_[i] + occ_per_irrep2[i]},
            Range{irrep_offsets_[i] + occ_per_irrep2[i],
                  irrep_offsets_[i] + irrep_sizes_[i]});
      } else if (i != j) {
        if (occ_per_irrep1[i] != 0 && occ_per_irrep1[i] != irrep_sizes_[i] &&
            occ_per_irrep2[j] != 0 && occ_per_irrep2[j] != irrep_sizes_[j]) {
          teit.tile(i, i, j, j) =
              temp2.tile(i, i, j, j)(Range{0, occ_per_irrep1[i]},
                                     Range{occ_per_irrep1[i], irrep_sizes_[i]},
                                     Range{0, occ_per_irrep2[j]},
                                     Range{occ_per_irrep2[j], irrep_sizes_[j]});
          denominator.tile(i, i, j, j) = full_denom(
              Range{irrep_offsets_[i], irrep_offsets_[i] + occ_per_irrep1[i]},
              Range{irrep_offsets_[i] + occ_per_irrep1[i],
                    irrep_offsets_[i] + irrep_sizes_[i]},
              Range{irrep_offsets_[j], irrep_offsets_[j] + occ_per_irrep2[j]},
              Range{irrep_offsets_[j] + occ_per_irrep2[j],
                    irrep_offsets_[j] + irrep_sizes_[j]});
        }

        if (occ_per_irrep1[i] != 0 && occ_per_irrep1[j] != irrep_sizes_[j] &&
            occ_per_irrep2[i] != 0 && occ_per_irrep2[j] != irrep_sizes_[j]) {
          teit.tile(i, j, i, j) =
              temp2.tile(i, j, i, j)(Range{0, occ_per_irrep1[i]},
                                     Range{occ_per_irrep1[j], irrep_sizes_[j]},
                                     Range{0, occ_per_irrep2[i]},
                                     Range{occ_per_irrep2[j], irrep_sizes_[j]});
          denominator.tile(i, j, i, j) = full_denom(
              Range{irrep_offsets_[i], irrep_offsets_[i] + occ_per_irrep1[i]},
              Range{irrep_offsets_[j] + occ_per_irrep1[j],
                    irrep_offsets_[j] + irrep_sizes_[j]},
              Range{irrep_offsets_[i], irrep_offsets_[i] + occ_per_irrep2[i]},
              Range{irrep_offsets_[j] + occ_per_irrep2[j],
                    irrep_offsets_[j] + irrep_sizes_[j]});
        }

        if (occ_per_irrep1[i] != 0 && occ_per_irrep2[i] != irrep_sizes_[i] &&
            occ_per_irrep2[j] != 0 && occ_per_irrep1[j] != irrep_sizes_[j]) {
          teit.tile(i, j, j, i) =
              temp2.tile(i, j, j, i)(Range{0, occ_per_irrep1[i]},
                                     Range{occ_per_irrep1[j], irrep_sizes_[j]},
                                     Range{0, occ_per_irrep2[j]},
                                     Range{occ_per_irrep2[i], irrep_sizes_[i]});
          denominator.tile(i, j, j, i) = full_denom(
              Range{irrep_offsets_[i], irrep_offsets_[i] + occ_per_irrep1[i]},
              Range{irrep_offsets_[j] + occ_per_irrep1[j],
                    irrep_offsets_[j] + irrep_sizes_[j]},
              Range{irrep_offsets_[j], irrep_offsets_[j] + occ_per_irrep2[j]},
              Range{irrep_offsets_[i] + occ_per_irrep2[i],
                    irrep_offsets_[i] + irrep_sizes_[i]});
        }

        if (occ_per_irrep2[i] != 0 && occ_per_irrep1[i] != irrep_sizes_[i] &&
            occ_per_irrep1[j] != 0 && occ_per_irrep2[j] != irrep_sizes_[j]) {
          teit.tile(j, i, i, j) =
              temp2.tile(j, i, i, j)(Range{0, occ_per_irrep1[j]},
                                     Range{occ_per_irrep1[i], irrep_sizes_[i]},
                                     Range{0, occ_per_irrep2[i]},
                                     Range{occ_per_irrep2[j], irrep_sizes_[j]});
          denominator.tile(j, i, i, j) = full_denom(
              Range{irrep_offsets_[j], irrep_offsets_[j] + occ_per_irrep1[j]},
              Range{irrep_offsets_[i] + occ_per_irrep1[i],
                    irrep_offsets_[i] + irrep_sizes_[i]},
              Range{irrep_offsets_[i], irrep_offsets_[i] + occ_per_irrep2[i]},
              Range{irrep_offsets_[j] + occ_per_irrep2[j],
                    irrep_offsets_[j] + irrep_sizes_[j]});
        }

        if (occ_per_irrep2[i] != irrep_sizes_[i] && occ_per_irrep1[j] != 0 &&
            occ_per_irrep1[i] != irrep_sizes_[i] && occ_per_irrep2[j] != 0) {
          teit.tile(j, i, j, i) =
              temp2.tile(j, i, j, i)(Range{0, occ_per_irrep1[j]},
                                     Range{occ_per_irrep1[i], irrep_sizes_[i]},
                                     Range{0, occ_per_irrep2[j]},
                                     Range{occ_per_irrep2[i], irrep_sizes_[i]});
          denominator.tile(j, i, j, i) = full_denom(
              Range{irrep_offsets_[j], irrep_offsets_[j] + occ_per_irrep1[j]},
              Range{irrep_offsets_[i] + occ_per_irrep1[i],
                    irrep_offsets_[i] + irrep_sizes_[i]},
              Range{irrep_offsets_[j], irrep_offsets_[j] + occ_per_irrep2[j]},
              Range{irrep_offsets_[i] + occ_per_irrep2[i],
                    irrep_offsets_[i] + irrep_sizes_[i]});
        }

        if (occ_per_irrep2[i] != 0 && occ_per_irrep2[i] != irrep_sizes_[i] &&
            occ_per_irrep1[j] != 0 && occ_per_irrep1[j] != irrep_sizes_[j]) {
          teit.tile(j, j, i, i) =
              temp2.tile(j, j, i, i)(Range{0, occ_per_irrep1[j]},
                                     Range{occ_per_irrep1[j], irrep_sizes_[j]},
                                     Range{0, occ_per_irrep2[i]},
                                     Range{occ_per_irrep2[i], irrep_sizes_[i]});
          denominator.tile(j, j, i, i) = full_denom(
              Range{irrep_offsets_[j], irrep_offsets_[j] + occ_per_irrep1[j]},
              Range{irrep_offsets_[j] + occ_per_irrep1[j],
                    irrep_offsets_[j] + irrep_sizes_[j]},
              Range{irrep_offsets_[i], irrep_offsets_[i] + occ_per_irrep2[i]},
              Range{irrep_offsets_[i] + occ_per_irrep2[i],
                    irrep_offsets_[i] + irrep_sizes_[i]});
        }
      }
    }
  }

  einsum(Indices{index::i, index::a, index::j, index::b}, &MP2_amps,
         Indices{index::i, index::a, index::j, index::b}, teit,
         Indices{index::i, index::a, index::j, index::b}, denominator);
}

void EinsumsUMP2::init_integrals() {
  // The basisset object contains all of the basis information and is formed in
  // the new_wavefunction call The integral factory oversees the creation of
  // integral objects
  auto integral = std::make_shared<IntegralFactory>(basisset_, basisset_,
                                                    basisset_, basisset_);

  tei_ = einsums::TiledTensor<double, 4>("TEI", irrep_sizes_);
  teitaa_ = einsums::TiledTensor<double, 4>("Transformed TEI (aa)",
                                            aocc_per_irrep_, aunocc_per_irrep_,
                                            aocc_per_irrep_, aunocc_per_irrep_);
  MP2_ampsaa_ = einsums::TiledTensor<double, 4>(
      "MP2 Amps (aa)", aocc_per_irrep_, aunocc_per_irrep_, aocc_per_irrep_,
      aunocc_per_irrep_);
  denominatoraa_ = TiledTensor<double, 4>("MP2 denominator (aa)",
                                          aocc_per_irrep_, aunocc_per_irrep_,
                                          aocc_per_irrep_, aunocc_per_irrep_);

  teitbb_ = einsums::TiledTensor<double, 4>("Transformed TEI (bb)",
                                            bocc_per_irrep_, bunocc_per_irrep_,
                                            bocc_per_irrep_, bunocc_per_irrep_);
  MP2_ampsbb_ = einsums::TiledTensor<double, 4>(
      "MP2 Amps (bb)", bocc_per_irrep_, bunocc_per_irrep_, bocc_per_irrep_,
      bunocc_per_irrep_);

  denominatorbb_ = TiledTensor<double, 4>("MP2 denominator (bb)",
                                          bocc_per_irrep_, bunocc_per_irrep_,
                                          bocc_per_irrep_, bunocc_per_irrep_);

  teitab_ = einsums::TiledTensor<double, 4>("Transformed TEI (ab)",
                                            aocc_per_irrep_, aunocc_per_irrep_,
                                            bocc_per_irrep_, bunocc_per_irrep_);
  MP2_ampsab_ = einsums::TiledTensor<double, 4>(
      "MP2 Amps (ab)", aocc_per_irrep_, aunocc_per_irrep_, bocc_per_irrep_,
      bunocc_per_irrep_);
  denominatorab_ = TiledTensor<double, 4>("MP2 denominator(ab)",
                                          aocc_per_irrep_, aunocc_per_irrep_,
                                          bocc_per_irrep_, bunocc_per_irrep_);

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

  timer_on("EinHF: Transforming Two-electron Integrals.");

#pragma omp taskgroup
  {
#pragma omp task depend(in : this->tei_, this->Ca_)                           \
    depend(out : this -> teitaa_, this->denominatoraa_, this->MP2_ampsaa_)
    { setup_spin_integrals(true, true); }
#pragma omp task depend(in : this->tei_, this->Cb_)                           \
    depend(out : this -> teitbb_, this->denominatorbb_, this->MP2_ampsbb_)
    { setup_spin_integrals(false, false); }
#pragma omp task depend(in : this->tei_, this->Ca_, this->Cb_)                \
    depend(out : this -> teitab_, this->denominatorab_, this->MP2_ampsab_)
    { setup_spin_integrals(true, false); }
  }

  timer_off("EinHF: Transforming Two-electron Integrals");
}

double EinsumsUMP2::compute_energy() {
  timer_on("EinHF: Computing MP2 energy");

  double e_new, eMP2_SS;

  Tensor<double, 0> eMP2_AA, eMP2_BB, eMP2_OS;

#pragma omp taskgroup
  {
#pragma omp task
    {
      einsum(0.0, Indices{}, &eMP2_OS, 1.0,
             Indices{index::i, index::a, index::j, index::b}, MP2_ampsab_,
             Indices{index::i, index::a, index::j, index::b}, teitab_);
    }
#pragma omp task
    {

      einsum(Indices{}, &eMP2_AA,
             Indices{index::i, index::a, index::j, index::b}, MP2_ampsaa_,
             Indices{index::i, index::a, index::j, index::b}, teitaa_);
      einsum(1.0, Indices{}, &eMP2_AA, -1.0,
             Indices{index::i, index::a, index::j, index::b}, MP2_ampsaa_,
             Indices{index::i, index::b, index::j, index::a}, teitaa_);
    }
#pragma omp task
    {

      einsum(Indices{}, &eMP2_BB,
             Indices{index::i, index::a, index::j, index::b}, MP2_ampsbb_,
             Indices{index::i, index::a, index::j, index::b}, teitbb_);
      einsum(1.0, Indices{}, &eMP2_BB, -1.0,
             Indices{index::i, index::a, index::j, index::b}, MP2_ampsbb_,
             Indices{index::i, index::b, index::j, index::a}, teitbb_);
    }
  }

  eMP2_SS = (double) eMP2_AA + (double) eMP2_BB;
  eMP2_SS /= 2.0;

  e_new = (double)eMP2_SS + (double)eMP2_OS;

  timer_off("EinHF: Computing energy");

  outfile->Printf("\tMP2 Same-spin:\t%lf\n", (double)eMP2_SS);
  outfile->Printf("\tMP2 Opposite-spin:\t%lf\n", (double)eMP2_OS);
  outfile->Printf("\tMP2 Correction:\t%lf\n", e_new);

  energy_ += e_new;
  outfile->Printf("\tTotal MP2 Energy:\t%lf\n", energy_);

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
