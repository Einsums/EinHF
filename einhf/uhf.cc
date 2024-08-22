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

#include "einsums.hpp"

#include "psi4/libfock/jk.h"
#include "psi4/libfock/v.h"
#include "psi4/libfunctional/superfunctional.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/mintshelper.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/pointgrp.h"
#include "psi4/libmints/sobasis.h"
#include "psi4/libmints/vector.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"
#include <LinearAlgebra.hpp>
#include <_Common.hpp>
#include <_Index.hpp>

using namespace einsums;

static std::string to_lower(const std::string &str) {
  std::string out(str);
  std::transform(str.begin(), str.end(), out.begin(),
                 [](char c) { return std::tolower(c); });
  return out;
}

namespace psi {
namespace einhf {

EinsumsUHF::EinsumsUHF(SharedWavefunction ref_wfn,
                       const std::shared_ptr<SuperFunctional> &functional,
                       Options &options)
    : Wavefunction(options) {

  timer_on("Setup Wavefunction");

  // Shallow copy useful objects from the passed in wavefunction
  shallow_copy(ref_wfn);

  print_ = options_.get_int("PRINT");

  nirrep_ = sobasisset_->nirrep();
  nso_ = basisset_->nbf();

  maxiter_ = options_.get_int("MAXITER");
  e_convergence_ = options_.get_double("E_CONVERGENCE");
  d_convergence_ = options_.get_double("D_CONVERGENCE");
  func_ = functional;

  print_header();
  if (options_.get_bool("DIIS")) {
    diis_max_iters_ = options_.get_int("DIIS_MAX_VECS");
  } else {
    diis_max_iters_ = 0;
    outfile->Printf("Turning DIIS off.\n");
  }

  init_integrals();

  evalsa_ = Tensor<double, 1>("Alpha orbital energies", nso_);
  evalsb_ = Tensor<double, 1>("Beta orbital energies", nso_);

  // Set Wavefunction matrices
  X_.set_name("S^1/2");
  Fa_.set_name("Alpha Fock Matrix");
  Fta_.set_name("Transformed Alpha Fock Matrix");
  Ca_.set_name("Alpha MO Coefficients");
  Cocca_.set_name("Occupied Alpha MO Coefficients");
  Da_.set_name("Alpha Density Matrix");
  JKwKa_.set_name("Alpha Two-electron Matrix");

  Fb_.set_name("Beta Fock Matrix");
  Ftb_.set_name("Transformed Beta Fock Matrix");
  Cb_.set_name("Beta MO Coefficients");
  Coccb_.set_name("Occupied Beta MO Coefficients");
  Db_.set_name("Beta Density Matrix");
  JKwKb_.set_name("Beta Two-electron Matrix");

  for (int i = 0; i < this->nirrep_; i++) {
    X_.push_block(einsums::Tensor<double, 2>(molecule_->irrep_labels().at(i),
                                             irrep_sizes_[i], irrep_sizes_[i]));
  }

  for (int i = 0; i < this->nirrep_; i++) {
    Fa_.push_block(einsums::Tensor<double, 2>(
        molecule_->irrep_labels().at(i), irrep_sizes_[i], irrep_sizes_[i]));
    Fb_.push_block(einsums::Tensor<double, 2>(
        molecule_->irrep_labels().at(i), irrep_sizes_[i], irrep_sizes_[i]));
  }

  for (int i = 0; i < this->nirrep_; i++) {
    Fta_.push_block(einsums::Tensor<double, 2>(
        molecule_->irrep_labels().at(i), irrep_sizes_[i], irrep_sizes_[i]));
    Ftb_.push_block(einsums::Tensor<double, 2>(
        molecule_->irrep_labels().at(i), irrep_sizes_[i], irrep_sizes_[i]));
  }

  for (int i = 0; i < this->nirrep_; i++) {
    Ca_.push_block(einsums::Tensor<double, 2>(
        molecule_->irrep_labels().at(i), irrep_sizes_[i], irrep_sizes_[i]));
    Cb_.push_block(einsums::Tensor<double, 2>(
        molecule_->irrep_labels().at(i), irrep_sizes_[i], irrep_sizes_[i]));
  }

  for (int i = 0; i < this->nirrep_; i++) {
    Cocca_.push_block(einsums::Tensor<double, 2>(
        molecule_->irrep_labels().at(i), irrep_sizes_[i], irrep_sizes_[i]));
    Coccb_.push_block(einsums::Tensor<double, 2>(
        molecule_->irrep_labels().at(i), irrep_sizes_[i], irrep_sizes_[i]));
  }

  for (int i = 0; i < this->nirrep_; i++) {
    Da_.push_block(einsums::Tensor<double, 2>(
        molecule_->irrep_labels().at(i), irrep_sizes_[i], irrep_sizes_[i]));
    Db_.push_block(einsums::Tensor<double, 2>(
        molecule_->irrep_labels().at(i), irrep_sizes_[i], irrep_sizes_[i]));
  }

  for (int i = 0; i < this->nirrep_; i++) {
    JKwKa_.push_block(einsums::Tensor<double, 2>(
        molecule_->irrep_labels().at(i), irrep_sizes_[i], irrep_sizes_[i]));
    JKwKb_.push_block(einsums::Tensor<double, 2>(
        molecule_->irrep_labels().at(i), irrep_sizes_[i], irrep_sizes_[i]));
  }
  timer_off("Setup Wavefunction");
}

EinsumsUHF::EinsumsUHF(const EinsumsUHF &ref_wfn)
    : Wavefunction(ref_wfn.options()) {
  print_ = ref_wfn.print_;
  naocc_ = ref_wfn.naocc_;
  nbocc_ = ref_wfn.nbocc_;
  aocc_per_irrep_ = ref_wfn.aocc_per_irrep_;
  bocc_per_irrep_ = ref_wfn.bocc_per_irrep_;
  irrep_sizes_ = ref_wfn.irrep_sizes_;
  nso_ = ref_wfn.nso_;
  maxiter_ = ref_wfn.maxiter_;
  diis_max_iters_ = ref_wfn.diis_max_iters_;
  e_nuc_ = ref_wfn.e_nuc_;
  d_convergence_ = ref_wfn.d_convergence_;
  e_convergence_ = ref_wfn.e_convergence_;
  H_ = ref_wfn.H_;
  S_ = ref_wfn.S_;
  X_ = ref_wfn.X_;

  Fa_ = ref_wfn.Fa_;
  JKwKa_ = ref_wfn.JKwKa_;
  Fta_ = ref_wfn.Fta_;
  Ca_ = ref_wfn.Ca_;
  Cocca_ = ref_wfn.Cocca_;
  Da_ = ref_wfn.Da_;
  evalsa_ = ref_wfn.evalsa_;
  Fb_ = ref_wfn.Fb_;
  JKwKb_ = ref_wfn.JKwKb_;
  Ftb_ = ref_wfn.Ftb_;
  Cb_ = ref_wfn.Cb_;
  Coccb_ = ref_wfn.Coccb_;
  Db_ = ref_wfn.Db_;
  evalsb_ = ref_wfn.evalsb_;
  jk_ = ref_wfn.jk_;
  func_ = ref_wfn.func_;
  v_ = ref_wfn.v_;
}

EinsumsUHF::EinsumsUHF(const EinsumsUHF &ref_wfn, Options &options)
    : Wavefunction(options) {
  print_ = ref_wfn.print_;
  naocc_ = ref_wfn.naocc_;
  nbocc_ = ref_wfn.nbocc_;
  aocc_per_irrep_ = ref_wfn.aocc_per_irrep_;
  bocc_per_irrep_ = ref_wfn.bocc_per_irrep_;
  irrep_sizes_ = ref_wfn.irrep_sizes_;
  nso_ = ref_wfn.nso_;
  maxiter_ = ref_wfn.maxiter_;
  diis_max_iters_ = ref_wfn.diis_max_iters_;
  e_nuc_ = ref_wfn.e_nuc_;
  d_convergence_ = ref_wfn.d_convergence_;
  e_convergence_ = ref_wfn.e_convergence_;
  H_ = ref_wfn.H_;
  S_ = ref_wfn.S_;
  X_ = ref_wfn.X_;

  Fa_ = ref_wfn.Fa_;
  JKwKa_ = ref_wfn.JKwKa_;
  Fta_ = ref_wfn.Fta_;
  Ca_ = ref_wfn.Ca_;
  Cocca_ = ref_wfn.Cocca_;
  Da_ = ref_wfn.Da_;
  evalsa_ = ref_wfn.evalsa_;
  Fb_ = ref_wfn.Fb_;
  JKwKb_ = ref_wfn.JKwKb_;
  Ftb_ = ref_wfn.Ftb_;
  Cb_ = ref_wfn.Cb_;
  Coccb_ = ref_wfn.Coccb_;
  Db_ = ref_wfn.Db_;
  evalsb_ = ref_wfn.evalsb_;
  jk_ = ref_wfn.jk_;
  func_ = ref_wfn.func_;
  v_ = ref_wfn.v_;
}

EinsumsUHF::~EinsumsUHF() {}

void EinsumsUHF::init_integrals() {
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
  nbocc_ = (nelec - molecule_->multiplicity() + 1) / 2;
  naocc_ = nbocc_ + molecule_->multiplicity() - 1;

  assert(naocc_ + nbocc_ == nelec);

  outfile->Printf("    There are %d alpha occupied orbitals and %d beta "
                  "occupied orbitals.\n",
                  naocc_, nbocc_);
  molecule_->print();
  if (print_ > 1) {
    basisset_->print_detail();
    options_.print();
  }

  // nso_ = basisset_->nso();

  // Nuclear repulsion without a field
  e_nuc_ = molecule_->nuclear_repulsion_energy({0, 0, 0});
  outfile->Printf("\n    Nuclear repulsion energy: %16.8f\n\n", e_nuc_);

  // Make a MintsHelper object to help
  auto mints = std::make_shared<MintsHelper>(basisset_);

  // These don't need to be declared, because they belong to the class
  auto S_mat = mints->so_overlap();

  // Core Hamiltonian is Kinetic + Potential
  auto H_mat = mints->so_kinetic();

  this->nirrep_ = (int)S_mat->nirrep();

  aocc_per_irrep_.resize(nirrep_);
  bocc_per_irrep_.resize(nirrep_);

  H_mat->add(mints->so_potential());

  H_.set_name("Hamiltonian");
  S_.set_name("Overlap");

  for (int i = 0; i < S_mat->nirrep(); i++) {
    irrep_sizes_.push_back(S_mat->rowdim(i));
    auto S_block = einsums::Tensor<double, 2>(
        molecule_->irrep_labels().at(i), S_mat->rowdim(i), S_mat->coldim(i));

    for (int j = 0; j < S_mat->rowdim(i); j++) {
      for (int k = 0; k < S_mat->coldim(i); k++) {
        S_block(j, k) = S_mat->get(i, j, k);
      }
    }

    S_.push_block(S_block);
  }

  for (int i = 0; i < S_mat->nirrep(); i++) {
    auto H_block = einsums::Tensor<double, 2>(
        molecule_->irrep_labels().at(i), H_mat->rowdim(i), H_mat->coldim(i));

    for (int j = 0; j < H_mat->rowdim(i); j++) {
      for (int k = 0; k < H_mat->coldim(i); k++) {
        H_block(j, k) = H_mat->get(i, j, k);
      }
    }

    H_.push_block(H_block);
  }

  if (print_ > 3) {
    fprintln(*outfile->stream(), S_);
    fprintln(*outfile->stream(), H_);
    outfile->stream()->flush();
  }

  outfile->Printf("    Forming JK object\n\n");

  size_t total_memory = Process::environment.get_memory() / 8 *
                        options_.get_double("SCF_MEM_SAFETY_FACTOR");

  // Construct a JK object that compute J and K SCF matrices very efficiently
  jk_ = JK::build_JK(basisset_, mintshelper_->get_basisset("DF_BASIS_SCF"),
                     options_, false, total_memory);

  jk_->set_do_K(func_->is_x_hybrid());
  jk_->set_do_wK(func_->is_x_lrc());

  jk_->set_omega(func_->x_omega());
  jk_->set_omega_alpha(func_->x_alpha());
  jk_->set_omega_beta(func_->x_beta());

  size_t jk_size = jk_->memory_estimate();

  if (jk_size < total_memory) {
    jk_->set_memory(jk_size);
  } else {
    jk_->set_memory(total_memory * 0.9);
  }

  // This is a very heavy compute object, lets give it 80% of our total memory
  // jk_->set_memory(Process::environment.get_memory() * 0.8);
  jk_->initialize();
  jk_->print_header();

  if (func_->needs_xc()) {
    v_ = VBase::build_V(basisset_, func_, options_, "UV");
    v_->initialize();
    v_->print_header();
  }
}

double EinsumsUHF::compute_electronic_energy(
    const einsums::BlockTensor<double, 2> &JKwK,
    const einsums::BlockTensor<double, 2> &D) {
  // Compute the electronic energy: (H + F)_pq * D_pq -> energy

  einsums::Tensor<double, 0> e_tens;
  auto temp = einsums::BlockTensor<double, 2>("temp", H_.vector_dims());

  temp = H_;
  temp *= 2;

  temp += JKwK;

  einsums::tensor_algebra::einsum(
      0.0, einsums::tensor_algebra::Indices{}, &e_tens, 1.0,
      einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                       einsums::tensor_algebra::index::j},
      D,
      einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                       einsums::tensor_algebra::index::j},
      temp);
  double out = (double)e_tens;

  if (func_->needs_xc()) {
    out += v_->quadrature_values()["FUNCTIONAL"];
  }

  if (func_->needs_vv10()) {
    out += v_->quadrature_values()["VV10"];
  }

  return out + scalar_variable("-D ENERGY");
}

void EinsumsUHF::update_Cocc(const einsums::Tensor<double, 1> &alpha_energies,
                             const einsums::Tensor<double, 1> &beta_energies) {
  // Update occupation.

  for (int i = 0; i < nirrep_; i++) {
    aocc_per_irrep_[i] = 0;
    bocc_per_irrep_[i] = 0;
  }

  // Alpha
  for (int i = 0; i < naocc_; i++) {
    double curr_min = INFINITY;
    int irrep_occ = -1;

    for (int j = 0; j < nirrep_; j++) {
      if (aocc_per_irrep_[j] >= irrep_sizes_[j]) {
        continue;
      }

      double energy = alpha_energies(S_.block_range(j)[0] + aocc_per_irrep_[j]);

      if (energy < curr_min) {
        curr_min = energy;
        irrep_occ = j;
      }
    }

    aocc_per_irrep_[irrep_occ]++;
  }

  // Beta
  for (int i = 0; i < nbocc_; i++) {
    double curr_min = INFINITY;
    int irrep_occ = -1;

    for (int j = 0; j < nirrep_; j++) {
      if (bocc_per_irrep_[j] >= irrep_sizes_[j]) {
        continue;
      }

      double energy = beta_energies(S_.block_range(j)[0] + bocc_per_irrep_[j]);

      if (energy < curr_min) {
        curr_min = energy;
        irrep_occ = j;
      }
    }

    bocc_per_irrep_[irrep_occ]++;
  }

  Cocca_.zero();
  Coccb_.zero();

// Alpha
#pragma omp parallel for
  for (int i = 0; i < nirrep_; i++) {
    Cocca_[i](einsums::AllT{}, einsums::Range(0, aocc_per_irrep_[i])) =
        Ca_[i](einsums::AllT{}, einsums::Range(0, aocc_per_irrep_[i]));
  }
// Beta
#pragma omp parallel for
  for (int i = 0; i < nirrep_; i++) {
    Coccb_[i](einsums::AllT{}, einsums::Range(0, bocc_per_irrep_[i])) =
        Cb_[i](einsums::AllT{}, einsums::Range(0, bocc_per_irrep_[i]));
  }
}

void EinsumsUHF::compute_diis_coefs(
    const std::deque<einsums::BlockTensor<double, 2>> &errors,
    std::vector<double> *out) const {
  einsums::Tensor<double, 2> *B_mat = new einsums::Tensor<double, 2>(
      "DIIS error matrix", errors.size() + 1, errors.size() + 1);

  B_mat->zero();
  (*B_mat)(einsums::Range{errors.size(), errors.size() + 1},
           einsums::Range{0, errors.size()}) = 1.0;
  (*B_mat)(einsums::Range{0, errors.size()},
           einsums::Range{errors.size(), errors.size() + 1}) = 1.0;

#pragma omp parallel for
  for (int i = 0; i < errors.size(); i++) {
#pragma omp parallel for
    for (int j = 0; j <= i; j++) {
      (*B_mat)(i, j) = einsums::linear_algebra::dot(errors[i], errors[j]);
      (*B_mat)(j, i) = (*B_mat)(i, j);
    }
  }

  einsums::Tensor<double, 2> res_mat =
      einsums::Tensor<double, 2>("DIIS result matrix", 1, errors.size() + 1);

  res_mat.zero();
  res_mat(0, errors.size()) = 1.0;

  einsums::linear_algebra::gesv(B_mat, &res_mat);

  delete B_mat;

  out->resize(errors.size());

  for (int i = 0; i < errors.size(); i++) {
    out->at(i) = res_mat(0, i);
  }
}

void EinsumsUHF::compute_diis_fock(
    const std::vector<double> &coefs,
    const std::deque<einsums::BlockTensor<double, 2>> &focks,
    einsums::BlockTensor<double, 2> *out) const {

  out->zero();

  for (int i = 0; i < coefs.size(); i++) {
    einsums::linear_algebra::axpy(coefs[i], focks[i], out);
  }
}

double EinsumsUHF::compute_energy() {
  if (diis_max_iters_ != 0) {
    outfile->Printf("Performing DIIS with %d vectors.\n", diis_max_iters_);
  } else {
    outfile->Printf("Turning DIIS off.\n");
  }

  // Allocate a few temporary matrices
  auto Temp1a = new einsums::BlockTensor<double, 2>("Alpha Temporary Array 1",
                                                    irrep_sizes_);
  auto Temp2a = new einsums::BlockTensor<double, 2>("Alpha Temporary Array 2",
                                                    irrep_sizes_);
  auto Temp1b = new einsums::BlockTensor<double, 2>("Beta Temporary Array 1",
                                                    irrep_sizes_);
  auto Temp2b = new einsums::BlockTensor<double, 2>("Beta Temporary Array 2",
                                                    irrep_sizes_);
  auto FDSa = new einsums::BlockTensor<double, 2>("Alpha FDS", irrep_sizes_);
  auto SDFa = new einsums::BlockTensor<double, 2>("Alpha SDF", irrep_sizes_);
  auto FDSb = new einsums::BlockTensor<double, 2>("Beta FDS", irrep_sizes_);
  auto SDFb = new einsums::BlockTensor<double, 2>("Beta SDF", irrep_sizes_);
  auto Evecsa =
      new einsums::BlockTensor<double, 2>("AlphaEigenvectors", irrep_sizes_);
  auto Evecsb =
      new einsums::BlockTensor<double, 2>("BetaEigenvectors", irrep_sizes_);
  auto Ja = new einsums::BlockTensor<double, 2>("Alpha J matrix", irrep_sizes_);
  auto Ka = new einsums::BlockTensor<double, 2>("Alpha K matrix", irrep_sizes_);
  auto wKa =
      new einsums::BlockTensor<double, 2>("Alpha wK matrix", irrep_sizes_);
  auto Va = new einsums::BlockTensor<double, 2>("Alpha V matrix", irrep_sizes_);

  auto Jb = new einsums::BlockTensor<double, 2>("Beta J matrix", irrep_sizes_);
  auto Kb = new einsums::BlockTensor<double, 2>("Beta K matrix", irrep_sizes_);
  auto wKb =
      new einsums::BlockTensor<double, 2>("Beta wK matrix", irrep_sizes_);
  auto Vb = new einsums::BlockTensor<double, 2>("Beta V matrix", irrep_sizes_);

  std::deque<einsums::BlockTensor<double, 2>>
      *errors = new std::deque<einsums::BlockTensor<double, 2>>(0),
      *focksa = new std::deque<einsums::BlockTensor<double, 2>>(0),
      *focksb = new std::deque<einsums::BlockTensor<double, 2>>(0);
  std::vector<double> *coefs = new std::vector<double>(0),
                      *error_vals = new std::vector<double>(0);

  std::vector<int> old_occsa, old_occsb;

  einsums::Tensor<double, 0> *dRMS_tensa = new einsums::Tensor<double, 0>(),
                             *dRMS_tensb = new einsums::Tensor<double, 0>();

  double *elec_a = new double(), *elec_b = new double();

// Form the X_ matrix (S^-1/2)
#pragma omp task depend(in : this->S_) depend(out : this -> X_)
  {
    timer_on("Form X");
    X_ = einsums::linear_algebra::pow(S_, -0.5);
    timer_off("Form X");
  }

#pragma omp task depend(in : this->H_) depend(out : this -> Fa_)
  {
    timer_on("Form Fa");
    Fa_ = H_;
    timer_off("Form Fa");
  }

#pragma omp task depend(in : this->H_) depend(out : this -> Fb_)
  {
    timer_on("Form Fb");
    Fb_ = H_;
    timer_off("Form Fb");
  }

#pragma omp task depend(inout : this->JKwKa_)
  { JKwKa_.zero(); }

#pragma omp task depend(inout : this->JKwKb_)
  { JKwKb_.zero(); }

// Alpha
#pragma omp task depend(in : this->Fa_, this->X_)                              \
    depend(out : *Temp1a, this -> Fta_, this->Ca_, this->evalsa_, *Evecsa)
  {
    timer_on("Form Ca");
    einsums::linear_algebra::gemm<false, false>(1.0, Fa_, X_, 0.0, Temp1a);
    einsums::linear_algebra::gemm<true, false>(1.0, X_, *Temp1a, 0.0, &Fta_);

    *Evecsa = Fta_;

    einsums::linear_algebra::syev(Evecsa, &evalsa_);

    einsums::linear_algebra::gemm<false, true>(1.0, X_, *Evecsa, 0.0, &Ca_);
    timer_off("Form Ca");
  }

// Beta
#pragma omp task depend(in : this->Fb_, this->X_)                              \
    depend(out : *Temp1b, this -> Ftb_, this->Cb_, this->evalsb_, *Evecsb)
  {
    timer_on("Form Cb");
    einsums::linear_algebra::gemm<false, false>(1.0, Fb_, X_, 0.0, Temp1b);
    einsums::linear_algebra::gemm<true, false>(1.0, X_, *Temp1b, 0.0, &Ftb_);

    *Evecsb = Ftb_;

    einsums::linear_algebra::syev(Evecsb, &evalsb_);

    einsums::linear_algebra::gemm<false, true>(1.0, X_, *Evecsb, 0.0, &Cb_);
    timer_off("Form Cb");
  }
#pragma omp taskwait depend(in : this->evalsa_, this->evalsb_, this->Ca_,      \
                                this->Cb_)                                     \
    depend(out : this -> Cocca_, this->Coccb_)

  // Update Cocc.
  update_Cocc(evalsa_, evalsb_);

#pragma omp task depend(in : this->Cocca_) depend(out : this -> Da_)
  {
    timer_on("Form Da");
    einsums::tensor_algebra::einsum(
        einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                         einsums::tensor_algebra::index::j},
        &Da_,
        einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                         einsums::tensor_algebra::index::m},
        Cocca_,
        einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::j,
                                         einsums::tensor_algebra::index::m},
        Cocca_);
    timer_off("Form Da");
  }

#pragma omp task depend(in : this->Coccb_) depend(out : this -> Db_)
  {
    timer_on("Form Db");
    einsums::tensor_algebra::einsum(
        einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                         einsums::tensor_algebra::index::j},
        &Db_,
        einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                         einsums::tensor_algebra::index::m},
        Coccb_,
        einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::j,
                                         einsums::tensor_algebra::index::m},
        Coccb_);
    timer_off("Form Db");
  }

  if (print_ > 3) {
#pragma omp taskwait depend(in : this->X_, this->Ca_, this->Da_,               \
                                this->evalsa_, this->Cocca_, this->Cb_,        \
                                this->Db_, this->evalsb_, this->Coccb_)
    outfile->Printf(
        "MO Coefficients and density from Core Hamiltonian guess:\n");
    fprintln(*outfile->stream(), X_);
    fprintln(*outfile->stream(), Ca_);
    fprintln(*outfile->stream(), Da_);
    fprintln(*outfile->stream(), evalsa_);
    fprintln(*outfile->stream(), Cocca_);
    fprintln(*outfile->stream(), Cb_);
    fprintln(*outfile->stream(), Db_);
    fprintln(*outfile->stream(), evalsb_);
    fprintln(*outfile->stream(), Coccb_);
  }

  int iter = 1;
  bool converged = false;
  double e_old;
  double e_new = e_nuc_;

  timer_on("Compute electronic energy");
#pragma omp task depend(in : this->H_, this->JKwKa_, this->Da_)                \
    depend(out : *elec_a)
  { *elec_a = compute_electronic_energy(JKwKa_, Da_); }
#pragma omp task depend(in : this->H_, this->JKwKb_, this->Db_)                \
    depend(out : *elec_b)
  { *elec_b = compute_electronic_energy(JKwKb_, Db_); }
#pragma omp taskwait depend(in : *elec_a, *elec_b)

  e_new += (*elec_a + *elec_b) / 2;
  timer_off("Compute electronic energy");

  outfile->Printf("    Energy from core Hamiltonian guess: %20.16f\n\n", e_new);

  old_occsa = aocc_per_irrep_;
  old_occsb = bocc_per_irrep_;

  outfile->Printf("    Initial Occupation:\n         \t");

  for (int i = 0; i < S_.num_blocks(); i++) {
    outfile->Printf("%4s\t", S_.name(i).c_str());
  }

  outfile->Printf("\n    AOCC \t");
  for (int i = 0; i < aocc_per_irrep_.size(); i++) {
    outfile->Printf("%4d\t", aocc_per_irrep_[i]);
  }

  outfile->Printf("\n    BOCC \t");
  for (int i = 0; i < bocc_per_irrep_.size(); i++) {
    outfile->Printf("%4d\t", bocc_per_irrep_[i]);
  }

  outfile->Printf("\n    VIRT \t");

  for (int i = 0; i < aocc_per_irrep_.size(); i++) {
    outfile->Printf("%4d\t", irrep_sizes_[i] - aocc_per_irrep_[i]);
  }

  outfile->Printf("\n    Total\t");

  for (int i = 0; i < aocc_per_irrep_.size(); i++) {
    outfile->Printf("%4d\t", irrep_sizes_[i]);
  }

  outfile->Printf("\n");

  outfile->Printf(
      "    *===========================================================*\n");
  outfile->Printf(
      "    * Iter       Energy            delta E    ||gradient||      *\n");
  outfile->Printf(
      "    *-----------------------------------------------------------*\n");

  while (!converged && iter < maxiter_) {
    e_old = e_new;

// Add the core Hamiltonian term to the Fock operator
#pragma omp task depend(in : this->H_) depend(out : this -> Fa_)
    {
      timer_on("Form Fa");
      Fa_ = H_;
    }

#pragma omp task depend(in : this->H_) depend(out : this -> Fb_)
    {
      timer_on("Form Fb");
      Fb_ = H_;
    }

#pragma omp task depend(inout : this->JKwKa_)
    { JKwKa_.zero(); }

#pragma omp task depend(inout : this->JKwKb_)
    { JKwKb_.zero(); }

    // The JK object handles all of the two electron integrals
    // To enhance efficiency it does use the density, but the orbitals
    // themselves D_uv = C_ui C_vj J_uv = I_uvrs D_rs K_uv = I_urvs D_rs

    // Here we clear the old Cocc and push_back our new one
    std::vector<SharedMatrix> &Cl = jk_->C_left();
    Cl.clear();

    SharedMatrix CTempa = std::make_shared<Matrix>(nirrep_, irrep_sizes_.data(),
                                                   aocc_per_irrep_.data());
    SharedMatrix CTempb = std::make_shared<Matrix>(nirrep_, irrep_sizes_.data(),
                                                   bocc_per_irrep_.data());
// Alpha
#pragma omp parallel for
    for (int i = 0; i < nirrep_; i++) {
      if (irrep_sizes_[i] == 0) {
        continue;
      }
      for (int j = 0; j < irrep_sizes_[i]; j++) {
        for (int k = 0; k < aocc_per_irrep_[i]; k++) {
          (*CTempa.get())(i, j, k) = Cocca_[i](j, k);
        }
      }
    }

// Beta
#pragma omp parallel for
    for (int i = 0; i < nirrep_; i++) {
      if (irrep_sizes_[i] == 0) {
        continue;
      }
      for (int j = 0; j < irrep_sizes_[i]; j++) {
        for (int k = 0; k < bocc_per_irrep_[i]; k++) {
          (*CTempb.get())(i, j, k) = Coccb_[i](j, k);
        }
      }
    }

    Cl.push_back(CTempa);
    Cl.push_back(CTempb);
    jk_->compute();

#pragma omp task depend(out : *Ja)
    {
      timer_on("Form Ja");
      const std::vector<SharedMatrix> &J_mat = jk_->J();
      for (int i = 0; i < nirrep_; i++) {
        if (irrep_sizes_[i] == 0) {
          continue;
        }
        for (int j = 0; j < irrep_sizes_[i]; j++) {
          for (int k = 0; k < irrep_sizes_[i]; k++) {
            (*Ja)[i](j, k) = J_mat[0]->get(i, j, k);
          }
        }
      }
      timer_off("Form Ja");
    }

#pragma omp task depend(out : *Jb)
    {
      timer_on("Form Jb");
      const std::vector<SharedMatrix> &J_mat = jk_->J();
      for (int i = 0; i < nirrep_; i++) {
        if (irrep_sizes_[i] == 0) {
          continue;
        }
        for (int j = 0; j < irrep_sizes_[i]; j++) {
          for (int k = 0; k < irrep_sizes_[i]; k++) {
            (*Jb)[i](j, k) = J_mat[1]->get(i, j, k);
          }
        }
      }
      timer_off("Form Jb");
    }

    if (func_->is_x_hybrid() && !(func_->is_x_lrc() && jk_->get_wcombine())) {
#pragma omp task depend(out : *Ka)
      {
        const std::vector<SharedMatrix> &K_mat = jk_->K();
        double alpha = func_->x_alpha();
        timer_on("Form Ka");
#pragma omp parallel for
        for (int i = 0; i < nirrep_; i++) {
          if (irrep_sizes_[i] == 0) {
            continue;
          }
          for (int j = 0; j < irrep_sizes_[i]; j++) {
            for (int k = 0; k < irrep_sizes_[i]; k++) {
              (*Ka)[i](j, k) = alpha * K_mat[0]->get(i, j, k);
            }
          }
        }
        timer_off("Form Ka");
      }

#pragma omp task depend(out : *Kb)
      {
        const std::vector<SharedMatrix> &K_mat = jk_->K();
        double alpha = func_->x_alpha();
        timer_on("Form Kb");
#pragma omp parallel for
        for (int i = 0; i < nirrep_; i++) {
          if (irrep_sizes_[i] == 0) {
            continue;
          }
          for (int j = 0; j < irrep_sizes_[i]; j++) {
            for (int k = 0; k < irrep_sizes_[i]; k++) {
              (*Kb)[i](j, k) = alpha * K_mat[1]->get(i, j, k);
            }
          }
        }
        timer_off("Form Kb");
      }
    }

    if (func_->is_x_lrc()) {
#pragma omp task depend(out : *wKa)
      {
        const std::vector<SharedMatrix> &wK_mat = jk_->wK();
        double beta = func_->x_beta();
        timer_on("Form wKa");
#pragma omp parallel for
        for (int i = 0; i < nirrep_; i++) {
          if (irrep_sizes_[i] == 0) {
            continue;
          }
          for (int j = 0; j < irrep_sizes_[i]; j++) {
            for (int k = 0; k < irrep_sizes_[i]; k++) {
              (*wKa)[i](j, k) = beta * wK_mat[0]->get(i, j, k);
            }
          }
        }
        timer_off("Form wKa");
      }

#pragma omp task depend(out : *wKb)
      {
        const std::vector<SharedMatrix> &wK_mat = jk_->wK();
        double beta = func_->x_beta();
        timer_on("Form wKb");
#pragma omp parallel for
        for (int i = 0; i < nirrep_; i++) {
          if (irrep_sizes_[i] == 0) {
            continue;
          }
          for (int j = 0; j < irrep_sizes_[i]; j++) {
            for (int k = 0; k < irrep_sizes_[i]; k++) {
              (*wKb)[i](j, k) = beta * wK_mat[1]->get(i, j, k);
            }
          }
        }
        timer_off("Form wKb");
      }
    }

    if (func_->needs_xc()) {

      SharedMatrix Da_mat = std::make_shared<Matrix>(
                       nirrep_, irrep_sizes_.data(), irrep_sizes_.data()),
                   Va_mat = std::make_shared<Matrix>(
                       nirrep_, irrep_sizes_.data(), irrep_sizes_.data());
      SharedMatrix Db_mat = std::make_shared<Matrix>(
                       nirrep_, irrep_sizes_.data(), irrep_sizes_.data()),
                   Vb_mat = std::make_shared<Matrix>(
                       nirrep_, irrep_sizes_.data(), irrep_sizes_.data());

      for (int i = 0; i < nirrep_; i++) {
        if (irrep_sizes_[i] == 0) {
          continue;
        }
        for (int j = 0; j < irrep_sizes_[i]; j++) {
          for (int k = 0; k < irrep_sizes_[i]; k++) {
            (*Da_mat.get())(i, j, k) = Da_[i](j, k);
          }
        }
      }

      for (int i = 0; i < nirrep_; i++) {
        if (irrep_sizes_[i] == 0) {
          continue;
        }
        for (int j = 0; j < irrep_sizes_[i]; j++) {
          for (int k = 0; k < irrep_sizes_[i]; k++) {
            (*Db_mat.get())(i, j, k) = Db_[i](j, k);
          }
        }
      }

      std::vector<SharedMatrix> D_vec{Da_mat, Db_mat};
      std::vector<SharedMatrix> V_vec{Va_mat, Vb_mat};

      v_->set_D(D_vec);
      v_->compute_V(V_vec);

#pragma omp task depend(out : *Va)
      {
        for (int i = 0; i < nirrep_; i++) {
          if (irrep_sizes_[i] == 0) {
            continue;
          }
          for (int j = 0; j < irrep_sizes_[i]; j++) {
            for (int k = 0; k < irrep_sizes_[i]; k++) {
              (*Va)[i](j, k) = Va_mat->get(i, j, k);
            }
          }
        }
      }

#pragma omp task depend(out : *Vb)
      {
        for (int i = 0; i < nirrep_; i++) {
          if (irrep_sizes_[i] == 0) {
            continue;
          }
          for (int j = 0; j < irrep_sizes_[i]; j++) {
            for (int k = 0; k < irrep_sizes_[i]; k++) {
              (*Vb)[i](j, k) = Vb_mat->get(i, j, k);
            }
          }
        }
      }
    }

#pragma omp task depend(in : *Ja, *Jb) depend(out : this -> Fa_, this->JKwKa_)
    {
      Fa_ += *Ja;
      Fa_ += *Jb;
      JKwKa_ += *Ja;
      JKwKa_ += *Jb;
    }
    if (func_->is_x_hybrid() && !(func_->is_x_lrc() && jk_->get_wcombine())) {
#pragma omp task depend(in : *Ka) depend(out : this -> Fa_, this->JKwKa_)
      {
        Fa_ -= *Ka;
        JKwKa_ -= *Ka;
      }
    }

    if (func_->is_x_lrc()) {
#pragma omp task depend(in : *wKa) depend(out : this -> Fa_, this->JKwKa_)
      {
        Fa_ -= *wKa;
        JKwKa_ -= *wKa;
      }
    }

    if (func_->needs_xc()) {
#pragma omp task depend(in : *Va) depend(out : this -> Fa_)
      { Fa_ += *Va; }
    }

#pragma omp task depend(in : *Ja, *Jb) depend(out : this -> Fb_, this->JKwKb_)
    {
      Fb_ += *Ja;
      Fb_ += *Jb;
      JKwKb_ += *Ja;
      JKwKb_ += *Jb;
    }
    if (func_->is_x_hybrid() && !(func_->is_x_lrc() && jk_->get_wcombine())) {
#pragma omp task depend(in : *Kb) depend(out : this -> Fb_, this->JKwKb_)
      {
        Fb_ -= *Kb;
        JKwKb_ -= *Kb;
      }
    }

    if (func_->is_x_lrc()) {
#pragma omp task depend(in : *wKb) depend(out : this -> Fb_, this->JKwKb_)
      {
        Fb_ -= *wKb;
        JKwKb_ -= *wKb;
      }
    }

    if (func_->needs_xc()) {
#pragma omp task depend(in : *Vb) depend(out : this -> Fb_)
      { Fb_ += *Vb; }
    }

#pragma omp task depend(in : this->Da_, this->S_, this->Fa_)                   \
    depend(out : *Temp1a, *FDSa)
    {
      timer_off("Form Fa");

      // Compute the orbital gradient, FDS-SDF
      einsums::linear_algebra::gemm<false, false>(1.0, Da_, S_, 0.0, Temp1a);
      einsums::linear_algebra::gemm<false, false>(1.0, Fa_, *Temp1a, 0.0, FDSa);
    }

#pragma omp task depend(in : this->Da_, this->S_, this->Fa_)                   \
    depend(out : *Temp2a, *SDFa)
    {
      einsums::linear_algebra::gemm<false, false>(1.0, Da_, Fa_, 0.0, Temp2a);
      einsums::linear_algebra::gemm<false, false>(1.0, S_, *Temp2a, 0.0, SDFa);
    }

#pragma omp task depend(in : this->Db_, this->S_, this->Fb_)                   \
    depend(out : *Temp1b, *FDSb)
    {
      timer_off("Form Fb");

      // Compute the orbital gradient, FDS-SDF
      einsums::linear_algebra::gemm<false, false>(1.0, Db_, S_, 0.0, Temp1b);
      einsums::linear_algebra::gemm<false, false>(1.0, Fb_, *Temp1b, 0.0, FDSb);
    }

#pragma omp task depend(in : this->Db_, this->S_, this->Fb_)                   \
    depend(out : *Temp2b, *SDFb)
    {
      einsums::linear_algebra::gemm<false, false>(1.0, Db_, Fb_, 0.0, Temp2b);
      einsums::linear_algebra::gemm<false, false>(1.0, S_, *Temp2b, 0.0, SDFb);
    }

#pragma omp task depend(in : *FDSa, *SDFa) depend(out : *Temp1a)
    {
      *Temp1a = *FDSa;
      *Temp1a -= *SDFa;
    }

#pragma omp task depend(in : *FDSb, *SDFb) depend(out : *Temp1b)
    {
      *Temp1b = *FDSb;
      *Temp1b -= *SDFb;
    }

    if (diis_max_iters_ > 0) {
#pragma omp taskwait depend(in : *Temp1a, *Temp1b)
      if (errors->size() == diis_max_iters_) {
        double max_error = -INFINITY;
        int max_ind = -1;

        for (int i = 0; i < diis_max_iters_; i++) {
          if (error_vals->at(i) > max_error) {
            max_error = error_vals->at(i);
            max_ind = i;
          }
        }

        focksa->at(max_ind) = Fa_;
        focksb->at(max_ind) = Fb_;
        errors->at(max_ind) = *Temp1a;
        errors->at(max_ind) += *Temp1b;
        error_vals->at(max_ind) = einsums::linear_algebra::dot(
            errors->at(max_ind), errors->at(max_ind));
      } else {
        errors->push_back(*Temp1a);
        errors->at(errors->size() - 1) += *Temp1b;
        focksa->push_back(Fa_);
        focksb->push_back(Fb_);
        error_vals->push_back(einsums::linear_algebra::dot(
            errors->at(errors->size() - 1), errors->at(errors->size() - 1)));
      }

      compute_diis_coefs(*errors, coefs);

      compute_diis_fock(*coefs, *focksa, &Fa_);
      compute_diis_fock(*coefs, *focksb, &Fb_);
    }

    // Density RMS
    *dRMS_tensa = 0;
    *dRMS_tensb = 0;

#pragma omp task depend(in : *Temp1a) depend(out : *dRMS_tensa)
    {
      einsums::tensor_algebra::einsum(
          0.0, einsums::tensor_algebra::Indices{}, dRMS_tensa,
          1.0 / (nso_ * nso_),
          einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                           einsums::tensor_algebra::index::j},
          *Temp1a,
          einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                           einsums::tensor_algebra::index::j},
          *Temp1a);
    }

#pragma omp task depend(in : *Temp1b) depend(out : *dRMS_tensb)
    {
      einsums::tensor_algebra::einsum(
          0.0, einsums::tensor_algebra::Indices{}, dRMS_tensb,
          1.0 / (nso_ * nso_),
          einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                           einsums::tensor_algebra::index::j},
          *Temp1b,
          einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                           einsums::tensor_algebra::index::j},
          *Temp1b);
    }

#pragma omp taskwait depend(in : *dRMS_tensa, *dRMS_tensb)
    double dRMS = std::sqrt((double)*dRMS_tensa + (double)*dRMS_tensb);

    // Compute the energy
    timer_on("Compute electronic energy");
#pragma omp task depend(in : this->H_, this->JKwKa_, this->Da_)                \
    depend(out : *elec_a)
    { *elec_a = compute_electronic_energy(JKwKa_, Da_); }
#pragma omp task depend(in : this->H_, this->JKwKb_, this->Db_)                \
    depend(out : *elec_b)
    { *elec_b = compute_electronic_energy(JKwKb_, Db_); }
#pragma omp taskwait depend(in : *elec_a, *elec_b)

    e_new = e_nuc_ + (*elec_a + *elec_b) / 2;
    timer_off("Compute electronic energy");
    double dE = e_new - e_old;

    converged = (fabs(dE) < e_convergence_) && (dRMS < d_convergence_);

    outfile->Printf("    * %3d %20.14f    %9.2e    %9.2e    ", iter, e_new, dE,
                    dRMS);
    if (focksa->size() > 0) {
      outfile->Printf("DIIS*\n");
    } else {
      outfile->Printf("    *\n");
    }

    // Alpha
#pragma omp task depend(in : this->Fa_, this->X_)                              \
    depend(out : *Temp1a, this -> Fta_, this->Ca_, this->evalsa_, *Evecsa)
    {
      timer_on("Form Ca");
      einsums::linear_algebra::gemm<false, false>(1.0, Fa_, X_, 0.0, Temp1a);
      einsums::linear_algebra::gemm<true, false>(1.0, X_, *Temp1a, 0.0, &Fta_);

      *Evecsa = Fta_;

      einsums::linear_algebra::syev(Evecsa, &evalsa_);

      einsums::linear_algebra::gemm<false, true>(1.0, X_, *Evecsa, 0.0, &Ca_);
      timer_off("Form Ca");
    }

// Beta
#pragma omp task depend(in : this->Fb_, this->X_)                              \
    depend(out : *Temp1b, this -> Ftb_, this->Cb_, this->evalsb_, *Evecsb)
    {
      timer_on("Form Cb");
      einsums::linear_algebra::gemm<false, false>(1.0, Fb_, X_, 0.0, Temp1b);
      einsums::linear_algebra::gemm<true, false>(1.0, X_, *Temp1b, 0.0, &Ftb_);

      *Evecsb = Ftb_;

      einsums::linear_algebra::syev(Evecsb, &evalsb_);

      einsums::linear_algebra::gemm<false, true>(1.0, X_, *Evecsb, 0.0, &Cb_);
      timer_off("Form Cb");
    }
#pragma omp taskwait depend(in : this->evalsa_, this->evalsb_, this->Ca_,      \
                                this->Cb_)                                     \
    depend(out : this -> Cocca_, this->Coccb_)

    // Update Cocc.
    update_Cocc(evalsa_, evalsb_);

    if (aocc_per_irrep_ != old_occsa || bocc_per_irrep_ != old_occsb) {
      outfile->Printf("    Occupation Changed:\n         \t");

      for (int i = 0; i < S_.num_blocks(); i++) {
        outfile->Printf("%4s\t", S_.name(i).c_str());
      }

      outfile->Printf("\n    AOCC \t");
      for (int i = 0; i < aocc_per_irrep_.size(); i++) {
        outfile->Printf("%4d\t", aocc_per_irrep_[i]);
      }

      outfile->Printf("\n    BOCC \t");
      for (int i = 0; i < bocc_per_irrep_.size(); i++) {
        outfile->Printf("%4d\t", bocc_per_irrep_[i]);
      }

      outfile->Printf("\n    VIRT \t");

      for (int i = 0; i < aocc_per_irrep_.size(); i++) {
        outfile->Printf("%4d\t", irrep_sizes_[i] - aocc_per_irrep_[i]);
      }

      outfile->Printf("\n    Total\t");

      for (int i = 0; i < aocc_per_irrep_.size(); i++) {
        outfile->Printf("%4d\t", irrep_sizes_[i]);
      }

      outfile->Printf("\n");
    }
    old_occsa = aocc_per_irrep_;
    old_occsb = bocc_per_irrep_;

#pragma omp task depend(in : this->Cocca_) depend(out : this -> Da_)
    {
      timer_on("Form Da");
      einsums::tensor_algebra::einsum(
          einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                           einsums::tensor_algebra::index::j},
          &Da_,
          einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                           einsums::tensor_algebra::index::m},
          Cocca_,
          einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::j,
                                           einsums::tensor_algebra::index::m},
          Cocca_);
      timer_off("Form Da");
    }

#pragma omp task depend(in : this->Coccb_) depend(out : this -> Db_)
    {
      timer_on("Form Db");
      einsums::tensor_algebra::einsum(
          einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                           einsums::tensor_algebra::index::j},
          &Db_,
          einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                           einsums::tensor_algebra::index::m},
          Coccb_,
          einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::j,
                                           einsums::tensor_algebra::index::m},
          Coccb_);
      timer_off("Form Db");
    }

#pragma omp taskwait depend(in : this->Da_, this->Db_)

    // Optional printing
    if (print_ > 3) {
#pragma omp taskwait depend(                                                   \
        in : this->Fta_, this->Fa_, *Evecsa, this->evalsa_, this->Ca_,         \
            this->Da_, *FDSa, *SDFa, *Temp1a, this->Ftb_, this->Fb_, *Evecsb,  \
            this->evalsb_, this->Cb_, this->Db_, *FDSb, *SDFb, *Temp1b)
      fprintln(*outfile->stream(), Fta_);
      fprintln(*outfile->stream(), Fa_);
      fprintln(*outfile->stream(), *Evecsa);
      fprintln(*outfile->stream(), evalsa_);
      fprintln(*outfile->stream(), Ca_);
      fprintln(*outfile->stream(), Da_);
      fprintln(*outfile->stream(), *FDSa);
      fprintln(*outfile->stream(), *SDFa);
      Temp1a->set_name("Orbital Gradient");
      fprintln(*outfile->stream(), *Temp1a);

      fprintln(*outfile->stream(), Ftb_);
      fprintln(*outfile->stream(), Fb_);
      fprintln(*outfile->stream(), *Evecsb);
      fprintln(*outfile->stream(), evalsb_);
      fprintln(*outfile->stream(), Cb_);
      fprintln(*outfile->stream(), Db_);
      fprintln(*outfile->stream(), *FDSb);
      fprintln(*outfile->stream(), *SDFb);
      Temp1b->set_name("Orbital Gradient");
      fprintln(*outfile->stream(), *Temp1b);
    }

    iter++;
  }
  outfile->Printf(
      "    *===========================================================*\n");

  if (!converged)
    throw PSIEXCEPTION("The SCF iterations did not converge.");

  // Compute S^2.
  einsums::Tensor<double, 0> spin;

  einsums::tensor_algebra::einsum(
      0.0, einsums::tensor_algebra::Indices{}, &spin, 0.5,
      einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                       einsums::tensor_algebra::index::j},
      Da_,
      einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                       einsums::tensor_algebra::index::j},
      Da_);

  einsums::tensor_algebra::einsum(
      1.0, einsums::tensor_algebra::Indices{}, &spin, -0.5,
      einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                       einsums::tensor_algebra::index::j},
      Db_,
      einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                       einsums::tensor_algebra::index::j},
      Db_);

  double s_squared = (double)spin * ((double)spin + 1);
  double s2_expected =
      (molecule_->multiplicity() + 1) * (molecule_->multiplicity() - 1) / 4.0;

  outfile->Printf("Spin contamination: %lf\nS^2 value: %lf\nS^2 expected: "
                  "%lf\nS expected: %lf\nS observed: %lf\n",
                  s_squared - s2_expected, s_squared, s2_expected,
                  (molecule_->multiplicity() - 1) / 2.0, (double)spin);

  evalsa_.set_name("Alpha Orbital Energies");
  outfile->Printf("\nAlpha Occupied:\n");

  std::vector<int> inds(nirrep_);
  for (int i = 0; i < nirrep_; i++) {
    inds[i] = 0;
  }

  for (int i = 0; i < naocc_; i++) {
    double curr_min = INFINITY;
    int min_ind = -1;

    for (int j = 0; j < nirrep_; j++) {
      if (inds[j] >= aocc_per_irrep_[j]) {
        continue;
      }
      if (evalsa_(S_.block_range(j)[0] + inds[j]) < curr_min) {
        curr_min = evalsa_(S_.block_range(j)[0] + inds[j]);
        min_ind = j;
      }
    }

    inds[min_ind]++;
    outfile->Printf("%d %s: %lf\n", inds[min_ind],
                    to_lower(S_[min_ind].name()).c_str(), curr_min);
  }

  outfile->Printf("Alpha Unoccupied:\n");

  for (int i = naocc_; i < nso_; i++) {
    double curr_min = INFINITY;
    int min_ind = -1;

    for (int j = 0; j < nirrep_; j++) {
      if (inds[j] >= irrep_sizes_[j]) {
        continue;
      }
      if (evalsa_(S_.block_range(j)[0] + inds[j]) < curr_min) {
        curr_min = evalsa_(S_.block_range(j)[0] + inds[j]);
        min_ind = j;
      }
    }

    inds[min_ind]++;
    outfile->Printf("%d %s: %lf\n", inds[min_ind],
                    to_lower(S_[min_ind].name()).c_str(), curr_min);
  }
  evalsb_.set_name("Beta Orbital Energies");

  outfile->Printf("\n\nBeta Occupied:\n");

  for (int i = 0; i < nirrep_; i++) {
    inds[i] = 0;
  }

  for (int i = 0; i < nbocc_; i++) {
    double curr_min = INFINITY;
    int min_ind = -1;

    for (int j = 0; j < nirrep_; j++) {
      if (inds[j] >= bocc_per_irrep_[j]) {
        continue;
      }
      if (evalsb_(S_.block_range(j)[0] + inds[j]) < curr_min) {
        curr_min = evalsb_(S_.block_range(j)[0] + inds[j]);
        min_ind = j;
      }
    }

    inds[min_ind]++;
    outfile->Printf("%d %s: %lf\n", inds[min_ind],
                    to_lower(S_[min_ind].name()).c_str(), curr_min);
  }

  outfile->Printf("Beta Unoccupied:\n");

  for (int i = nbocc_; i < nso_; i++) {
    double curr_min = INFINITY;
    int min_ind = -1;

    for (int j = 0; j < nirrep_; j++) {
      if (inds[j] >= irrep_sizes_[j]) {
        continue;
      }
      if (evalsb_(S_.block_range(j)[0] + inds[j]) < curr_min) {
        curr_min = evalsb_(S_.block_range(j)[0] + inds[j]);
        min_ind = j;
      }
    }

    inds[min_ind]++;
    outfile->Printf("%d %s: %lf\n", inds[min_ind],
                    to_lower(S_[min_ind].name()).c_str(), curr_min);
  }

  energy_ = e_new;

  delete coefs;
  delete focksa;
  delete errors;

  delete Temp1a;
  delete Temp2a;
  delete Evecsa;
  delete FDSa;
  delete SDFa;
  delete Ja;
  delete Ka;
  delete wKa;
  delete Va;

  delete focksb;

  delete Temp1b;
  delete Temp2b;
  delete Evecsb;
  delete FDSb;
  delete SDFb;
  delete Jb;
  delete Kb;
  delete wKb;
  delete Vb;

  delete dRMS_tensa;
  delete dRMS_tensb;

  delete error_vals;

  delete elec_a;
  delete elec_b;

  return e_new;
}
void EinsumsUHF::print_header() {
  int nthread = Process::environment.get_n_threads();

  outfile->Printf("\n");
  outfile->Printf(
      "         ---------------------------------------------------------\n");
  outfile->Printf("                                   Einsums SCF\n");
  outfile->Printf("                                by Connor Briggs\n");
  outfile->Printf("                                 %4s Reference\n",
                  options_.get_str("REFERENCE").c_str());
  outfile->Printf("                               Running on the CPU\n");
  outfile->Printf("                      %3d Threads, %6ld MiB Core\n", nthread,
                  memory_ / 1048576L);
  outfile->Printf(
      "         ---------------------------------------------------------\n");
  outfile->Printf("\n");
  outfile->Printf("  ==> Geometry <==\n\n");

  molecule_->print();

  outfile->Printf("  Running in %s symmetry.\n\n",
                  molecule_->point_group()->symbol().c_str());

  molecule_->print_rotational_constants();

  outfile->Printf("  Nuclear repulsion = %20.15f\n\n",
                  molecule_->nuclear_repulsion_energy({0, 0, 0}));
  outfile->Printf("  Charge       = %d\n", molecule_->molecular_charge());
  outfile->Printf("  Multiplicity = %d\n", molecule_->multiplicity());
  outfile->Printf("  Nalpha       = %d\n", nalpha_);
  outfile->Printf("  Nbeta        = %d\n\n", nbeta_);

  outfile->Printf("  ==> Algorithm <==\n\n");
  outfile->Printf("  SCF Algorithm Type is %s.\n",
                  options_.get_str("SCF_TYPE").c_str());
  outfile->Printf("  DIIS %s.\n",
                  options_.get_bool("DIIS") ? "enabled" : "disabled");
  outfile->Printf("  Energy threshold   = %3.2e\n",
                  options_.get_double("E_CONVERGENCE"));
  outfile->Printf("  Density threshold  = %3.2e\n",
                  options_.get_double("D_CONVERGENCE"));

  outfile->Printf("  ==> Primary Basis <==\n\n");

  basisset_->print_by_level("outfile", print_);

  outfile->Printf("  ==> Functional <==\n\n");
  func_->print("outfile", print_);
}
} // namespace einhf
} // namespace psi
