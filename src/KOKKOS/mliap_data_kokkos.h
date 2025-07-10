/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Matt Bettencourt (NVIDIA)
------------------------------------------------------------------------- */

#ifndef LMP_MLIAP_DATA_KOKKOS_H
#define LMP_MLIAP_DATA_KOKKOS_H

#include "mliap_data.h"

#include "kokkos_type.h"
#include "memory_kokkos.h"
#include "pair_mliap_kokkos.h"
#include "pointers.h"

namespace LAMMPS_NS {
// clang-format off
enum {
  IATOMS_MASK      = 0x00000001,
  IELEMS_MASK      = 0x00000002,
  JATOMS_MASK      = 0x00000004,
  JELEMS_MASK      = 0x00000008,
  IJ_MASK          = 0x00000010,
  BETAS_MASK       = 0x00000020,
  DESCRIPTORS_MASK = 0x00000040,
  EATOMS_MASK      = 0x00000080,
  RIJ_MASK         = 0x00000100,
  GRADFORCE_MASK   = 0x00000200,
  GRADDESC_MASK    = 0x00000400,
  NUMNEIGHS_MASK   = 0x00000800,
  GAMMA_MASK_MASK  = 0x00001000,
  GAMMA_ROW_MASK   = 0x00002000,
  GAMMA_COL_MASK   = 0x00004000,
  PAIR_I_MASK      = 0x00008000,
  ELEMS_MASK       = 0x00010000,
};
// clang-format on

#define MAX_NUM_EXTRA_PROPERTIES 10
#define MAX_NAME_LENGTH 256

template <class DeviceType> struct ExtraProperties {
  using execution_space = typename DeviceType::execution_space;
  using memory_space = typename DeviceType::memory_space;

  using View2D = Kokkos::View<double**, Kokkos::LayoutRight, DeviceType>;

  std::unordered_map<char*, View2D> data;  //property name
  std::unordered_map<char*, int> dims;     //property dim


  DAT::tdual_float_1d flat_data;
  DAT::tdual_int_1d   offsets;
  DAT::tdual_int_1d   dims;
  Kokkos::DualView<char**, DeviceType> names;
  int nproperties;
  int nmax;
  int max_num_elems;

  ExtraProperties() : nproperties(0), nmax(0), max_num_elems(0),
                      offsets("ExtraProperties:offsets", MAX_NUM_EXTRA_PROPERTIES),
                      dims("ExtraProperties:dims", MAX_NUM_EXTRA_PROPERTIES),
                      names("ExtraProperties:names", MAX_NUM_EXTRA_PROPERTIES, MAX_NAME_LENGTH) { }

  /*KOKKOS_INLINE_FUNCTION
  LMP_FLOAT& operator()(int i, int j, int k) {
    //TODO: Implement bounds checking?
    int base = offsets.d_view(i);
    int dim = dims.d_view(i);
    int index = j*dim + k;
    return flat_data.d_view(base + index);
  }*/

  KOKKOS_INLINE_FUNCTION
  LMP_FLOAT& operator()(const char* name, int i, int j) {
    return data[name](i, j);
  }

  /*KOKKOS_INLINE_FUNCTION
  int get_dim(int index) {
    if (index >= nproperties) {
      return -1;
    }
    return dims.d_view(index);
  }*/

  KOKKOS_INLINE_FUNCTION
  int get_dim(const char* name) const {
    auto i = dims.find(name);
    return (i != dims.end()) ? i->second : -1;
  }

  //KOKKOS_INLINE_FUNCTION
  //Kokkos::View<LMP_FLOAT**, Kokkos::LayoutStride, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
  //get_2d_view(int index) {
  //  int offset = offsets(index);
  //  int dim = dims(index);
  //  LMP_FLOAT* base = flat_data.data();
  //  LMP_FLOAT* ptr_to_data = base + offset;
  //  Kokkos::LayoutStride layout(nmax, dim, dim, 1);
  //  Kokkos::View<LMP_FLOAT**, Kokkos::LayoutStride, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
  //          retVal(ptr_to_data, layout);
  //  return retVal;
  //}

  LMP_FLOAT* get_data_pointer(int index) {
    int offset = offsets(index);
    LMP_FLOAT* base = flat_data.data();
    LMP_FLOAT* ptr_to_data = base + offset;
    return ptr_to_data;
  }

  int register_extra_property(char* name, int dim) {
    //Add name and dim to list
    names.h_view(nproperties) = name;
    dims.h_view(nproperties) = dim;
    modify_host(0);
    sync_device(0);
    nproperties++;
    //Grow flat_data
    grow(nmax);
    //Return index
    return nproperties - 1;
  }

  void grow(int new_nmax) {
    //Check new size, return if strictly smaller than nmax (equal to can be resizing for new props)
    if (new_nmax < nmax) return;
    //Compute new size and offsets array
    int new_total_elems = 0;
    for (int i = 0; i < nproperties; i++) {
      offsets.h_view(i) = new_total_elems;
      new_total_elems += new_namx * dims.h_view(i);
    }
    modify_host(0);
    sync_device(0);
    //Allocate a new view with new size
    DAT::tdual_float_1d new_view("resizedExtraProperties", new_total_elems);
    //Copy all existing elements to new view
    if (max_num_elems != 0) {
      auto newSubViewD = Kokkos::subview(new_view.d_view, std::make_pair(0, max_num_elems));
      auto newSubViewH = Kokkos::subview(new_view.h_view, std::make_pair(0, max_num_elems));
      Kokkos::deepcopy(newSubViewD, flat_data.d_view);
      Kokkos::deepcopy(newSubViewH, flat_data.h_view);
      flat_data = new_view;
    }
    //Finally assign new max_elems and nmax
    nmax = new_namx;
    num_max_elems = new_total_elems;
  }

  /*//Flat data only set to 1 will only mark flat_data has modified
  void modify_host(int flat_data_only = 1) {
    flat_data.modify<Kokkos::HostSpace>();
    if (flat_data_only != 1) {
      offsets.modify<Kokkos::HostSpace>();
      dims.modify<Kokkos::HostSpace>();
      names.modify<Kokkos::HostSpace>();
    }
  }

  void modify_device(int flat_data_only = 1) {
    flat_data.modify<execution_space>();
    if (flat_data_only != 1) {
      offsets.modify<execution_space>();
      dims.modify<execution_space>();
      names.modify<execution_space>();
    }
  }

  void sync_host(int flat_data_only = 1) {
    flat_data.sync<Kokkos::HostSpace>();
    if (flat_data_only != 1) {
      offsets.sync<Kokkos::HostSpace>();
      dims.sync<Kokkos::HostSpace>();
      names.sync<Kokkos::HostSpace>();
    }
  }

  void sync_device(int flat_data_only = 1) {
    flat_data.sync<execution_space>();
    if (flat_data_only != 1) {
      offsets.sync<execution_space>();
      dims.sync<execution_space>();
      names.sync<execution_space>();
    }
  }*/
}; 

template <class DeviceType> class MLIAPDataKokkos : public MLIAPData {
 public:
  MLIAPDataKokkos(class LAMMPS *, int, int *, class MLIAPModel *, class MLIAPDescriptor *,
                  class PairMLIAPKokkos<DeviceType> * = nullptr);
  ~MLIAPDataKokkos() override;
  ExecutionSpace execution_space;

  void generate_neighdata(class NeighList *, int = 0, int = 0) override;
  void grow_neigharrays() override;

  void modified(ExecutionSpace space, unsigned int mask, bool ignore_auto_sync = false);

  void sync(ExecutionSpace space, unsigned int mask, bool ignore_auto_sync = false);

  PairMLIAPKokkos<DeviceType> *k_pairmliap;

  DAT::tdual_int_1d k_iatoms;           // index of each atom
  DAT::tdual_int_1d k_ielems;           // element of each atom
  DAT::tdual_int_1d k_jatoms;           // index of each neighbor
  DAT::tdual_int_1d k_elems;            // element of each atom in or not in the neighborlist
  DAT::tdual_int_1d k_pair_i;           // index of each i atom for each ij pair
  DAT::tdual_int_1d k_jelems;           // element of each neighbor
  DAT::tdual_int_1d k_ij;               // Start location for each particle
  DAT::tdual_float_2d k_betas;          // betas for all atoms in list
  DAT::tdual_float_2d k_descriptors;    // descriptors for all atoms in list
  DAT::tdual_float_1d k_eatoms;         // energies for all atoms in list
  DAT::tdual_float_2d k_rij;            // distance vector of each neighbor
  DAT::tdual_float_2d k_gradforce;
  DAT::tdual_float_3d k_graddesc;         // descriptor gradient w.r.t. each neighbor
  DAT::tdual_int_1d k_numneighs;          // neighbors count for each atom
  DAT::tdual_float_2d k_gamma;            // gamma element
  DAT::tdual_int_2d k_gamma_row_index;    // row (parameter) index
  DAT::tdual_int_2d k_gamma_col_index;    // column (descriptor) index

  // Just cached for python interface
  double *f_device;

 protected:
  class LAMMPS *lmp;
};

// Now we need a specific device version for communication with python
class MLIAPDataKokkosDevice {
public:

  MLIAPDataKokkosDevice(MLIAPDataKokkos<LMPDeviceType> &base) :
    size_array_rows(base.size_array_rows),
    size_array_cols(base.size_array_cols),
    natoms(base.natoms),
    yoffset(base.yoffset),
    zoffset(base.zoffset),
    ndims_force(base.ndims_force),
    ndims_virial(base.ndims_virial),
    size_gradforce(base.size_gradforce),
    f(base.f_device),
    gradforce(base.k_gradforce.d_view.data()),
    betas(base.k_betas.d_view.data()),
    descriptors(base.k_descriptors.d_view.data()),
    eatoms(base.k_eatoms.d_view.data()),
    energy(&base.energy),
    ndescriptors(base.ndescriptors),
    nparams(base.nparams),
    nelements(base.nelements),
    gamma_nnz(base.gamma_nnz),
    gamma(base.k_gamma.d_view.data()),
    gamma_row_index(base.k_gamma_row_index.d_view.data()),
    gamma_col_index(base.k_gamma_col_index.d_view.data()),
    egradient(nullptr),
    ntotal(base.ntotal),
    nlistatoms(base.nlistatoms),
    nlocal(base.nlocal),
    natomneigh(base.natomneigh),
    numneighs(base.numneighs),
    iatoms(base.k_iatoms.d_view.data()),
    pair_i(base.k_pair_i.d_view.data()),
    ielems(base.k_ielems.d_view.data()),
    nneigh_max(base.nneigh_max),
    npairs(base.npairs),
    jatoms(base.k_jatoms.d_view.data()),
    jelems(base.k_jelems.d_view.data()),
    elems(base.k_elems.d_view.data()),
    rij(base.k_rij.d_view.data()),
    graddesc(base.k_graddesc.d_view.data()),
    eflag(base.eflag),
    vflag(base.vflag),
    pairmliap(dynamic_cast<PairMLIAPKokkos<LMPDeviceType> *>(base.pairmliap)),
#if defined(KOKKOS_ENABLE_CUDA)
    dev(1)
#else
    dev(0)
#endif
    {  }
  int size_array_rows;
  int size_array_cols;
  int natoms;
  int yoffset;
  int zoffset;
  int ndims_force;
  int ndims_virial;
  int size_gradforce;

  //Write only
  double *f;
  double *gradforce;
  double *betas;
  double *descriptors;
  double *eatoms;
  double *energy;

  // sizing
  const int ndescriptors;
  const int nparams;
  const int nelements;

  //Ignored for now
  int gamma_nnz;
  double *gamma;
  int *gamma_row_index;
  int *gamma_col_index;
  double *egradient;

  // Neighborlist stuff
  const int ntotal;
  const int nlistatoms;
  const int nlocal;
  const int natomneigh;
  int *numneighs;
  int *iatoms;
  int *pair_i;
  int *ielems;
  const int nneigh_max;
  const int npairs;
  int *jatoms;
  int *jelems;
  int *elems;
  double *rij;
  double *graddesc;
  int eflag;
  int vflag;

  class PairMLIAPKokkos<LMPDeviceType> *pairmliap;    // access to pair tally functions

  int dev;

  //forward_exchange writes into ghosts
  template <typename CommType>
  void forward_exchange(CommType* copy_from, CommType* copy_to, const int vec_len){
    pairmliap->forward_comm(copy_from, copy_to, vec_len);
  }

  //reverse_exchange adds from ghosts and zeros out ghosts afterwards
  template <typename CommType>
  void reverse_exchange(CommType* copy_from, CommType* copy_to, const int vec_len){
    pairmliap->reverse_comm(copy_from, copy_to, vec_len);
  }


#ifdef LMP_KOKKOS_GPU
  MLIAPDataKokkosDevice(MLIAPDataKokkos<LMPHostType> &base) : ndescriptors(-1),nparams(-1),nelements(-1),ntotal(-1),nlistatoms(-1),nlocal(-1),natomneigh(-1),
      nneigh_max(-1),npairs(-1)
  {
    // It cannot get here, but needed for compilation
  }
#endif
};


}    // namespace LAMMPS_NS
#endif
