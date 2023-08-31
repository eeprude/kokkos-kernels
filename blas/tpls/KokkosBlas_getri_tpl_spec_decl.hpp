//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

// AquiEEP

#ifndef KOKKOSBLAS_GETRI_TPL_SPEC_DECL_HPP_
#define KOKKOSBLAS_GETRI_TPL_SPEC_DECL_HPP_

namespace KokkosBlas {
namespace Impl {
template <class AViewType, class BViewType, class PViewType>
inline void getri_print_specialization() {
#ifdef KOKKOSKERNELS_ENABLE_CHECK_SPECIALIZATION
#ifdef KOKKOSKERNELS_ENABLE_TPL_MAGMA
  printf("KokkosBlas::getri<> TPL MAGMA specialization for < %s , %s, %s >\n",
         typeid(AViewType).name(), typeid(BViewType).name(),
         typeid(PViewType).name());
#else
#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS
  printf("KokkosBlas::getri<> TPL Blas specialization for < %s , %s, %s >\n",
         typeid(AViewType).name(), typeid(BViewType).name(),
         typeid(PViewType).name());
#endif
#endif
#endif
}
}  // namespace Impl
}  // namespace KokkosBlas

// Generic Host side BLAS (could be MKL or whatever)
#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS
#include <KokkosBlas_Host_tpl.hpp>

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS_DGETRI_BLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)              \
  template <class ExecSpace>                                                  \
  struct GETRI<                                                                \
      Kokkos::View<double**, LAYOUT, Kokkos::Device<ExecSpace, MEM_SPACE>,    \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<double**, LAYOUT, Kokkos::Device<ExecSpace, MEM_SPACE>,    \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<int*, LAYOUT,                                              \
                   Kokkos::Device<Kokkos::DefaultHostExecutionSpace,          \
                                  Kokkos::HostSpace>,                         \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      true, ETI_SPEC_AVAIL> {                                                 \
    typedef double SCALAR;                                                    \
    typedef Kokkos::View<SCALAR**, LAYOUT,                                    \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        AViewType;                                                            \
    typedef Kokkos::View<SCALAR**, LAYOUT,                                    \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        BViewType;                                                            \
    typedef Kokkos::View<                                                     \
        int*, LAYOUT,                                                         \
        Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace>, \
        Kokkos::MemoryTraits<Kokkos::Unmanaged> >                             \
        PViewType;                                                            \
                                                                              \
    static void getri(const AViewType& A, const BViewType& B,                  \
                     const PViewType& IPIV) {                                 \
      Kokkos::Profiling::pushRegion("KokkosBlas::getri[TPL_BLAS,double]");     \
      getri_print_specialization<AViewType, BViewType, PViewType>();           \
      const bool with_pivot =                                                 \
          !((IPIV.extent(0) == 0) && (IPIV.data() == nullptr));               \
                                                                              \
      const int N    = static_cast<int>(A.extent(1));                         \
      const int AST  = static_cast<int>(A.stride(1));                         \
      const int LDA  = (AST == 0) ? 1 : AST;                                  \
      const int BST  = static_cast<int>(B.stride(1));                         \
      const int LDB  = (BST == 0) ? 1 : BST;                                  \
      const int NRHS = static_cast<int>(B.extent(1));                         \
                                                                              \
      int info = 0;                                                           \
                                                                              \
      if (with_pivot) {                                                       \
        HostBlas<double>::getri(N, NRHS, A.data(), LDA, IPIV.data(), B.data(), \
                               LDB, info);                                    \
      }                                                                       \
      Kokkos::Profiling::popRegion();                                         \
    }                                                                         \
  };

#define KOKKOSBLAS_SGETRI_BLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)              \
  template <class ExecSpace>                                                  \
  struct GETRI<                                                                \
      Kokkos::View<float**, LAYOUT, Kokkos::Device<ExecSpace, MEM_SPACE>,     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<float**, LAYOUT, Kokkos::Device<ExecSpace, MEM_SPACE>,     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<int*, LAYOUT,                                              \
                   Kokkos::Device<Kokkos::DefaultHostExecutionSpace,          \
                                  Kokkos::HostSpace>,                         \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      true, ETI_SPEC_AVAIL> {                                                 \
    typedef float SCALAR;                                                     \
    typedef Kokkos::View<SCALAR**, LAYOUT,                                    \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        AViewType;                                                            \
    typedef Kokkos::View<SCALAR**, LAYOUT,                                    \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        BViewType;                                                            \
    typedef Kokkos::View<                                                     \
        int*, LAYOUT,                                                         \
        Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace>, \
        Kokkos::MemoryTraits<Kokkos::Unmanaged> >                             \
        PViewType;                                                            \
                                                                              \
    static void getri(const AViewType& A, const BViewType& B,                  \
                     const PViewType& IPIV) {                                 \
      Kokkos::Profiling::pushRegion("KokkosBlas::getri[TPL_BLAS,float]");      \
      getri_print_specialization<AViewType, BViewType, PViewType>();           \
      const bool with_pivot =                                                 \
          !((IPIV.extent(0) == 0) && (IPIV.data() == nullptr));               \
                                                                              \
      const int N    = static_cast<int>(A.extent(1));                         \
      const int AST  = static_cast<int>(A.stride(1));                         \
      const int LDA  = (AST == 0) ? 1 : AST;                                  \
      const int BST  = static_cast<int>(B.stride(1));                         \
      const int LDB  = (BST == 0) ? 1 : BST;                                  \
      const int NRHS = static_cast<int>(B.extent(1));                         \
                                                                              \
      int info = 0;                                                           \
                                                                              \
      if (with_pivot) {                                                       \
        HostBlas<float>::getri(N, NRHS, A.data(), LDA, IPIV.data(), B.data(),  \
                              LDB, info);                                     \
      }                                                                       \
      Kokkos::Profiling::popRegion();                                         \
    }                                                                         \
  };

#define KOKKOSBLAS_ZGETRI_BLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)              \
  template <class ExecSpace>                                                  \
  struct GETRI<Kokkos::View<Kokkos::complex<double>**, LAYOUT,                 \
                           Kokkos::Device<ExecSpace, MEM_SPACE>,              \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,         \
              Kokkos::View<Kokkos::complex<double>**, LAYOUT,                 \
                           Kokkos::Device<ExecSpace, MEM_SPACE>,              \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,         \
              Kokkos::View<int*, LAYOUT,                                      \
                           Kokkos::Device<Kokkos::DefaultHostExecutionSpace,  \
                                          Kokkos::HostSpace>,                 \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,         \
              true, ETI_SPEC_AVAIL> {                                         \
    typedef Kokkos::complex<double> SCALAR;                                   \
    typedef Kokkos::View<SCALAR**, LAYOUT,                                    \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        AViewType;                                                            \
    typedef Kokkos::View<SCALAR**, LAYOUT,                                    \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        BViewType;                                                            \
    typedef Kokkos::View<                                                     \
        int*, LAYOUT,                                                         \
        Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace>, \
        Kokkos::MemoryTraits<Kokkos::Unmanaged> >                             \
        PViewType;                                                            \
                                                                              \
    static void getri(const AViewType& A, const BViewType& B,                  \
                     const PViewType& IPIV) {                                 \
      Kokkos::Profiling::pushRegion(                                          \
          "KokkosBlas::getri[TPL_BLAS,complex<double>]");                      \
      getri_print_specialization<AViewType, BViewType, PViewType>();           \
      const bool with_pivot =                                                 \
          !((IPIV.extent(0) == 0) && (IPIV.data() == nullptr));               \
                                                                              \
      const int N    = static_cast<int>(A.extent(1));                         \
      const int AST  = static_cast<int>(A.stride(1));                         \
      const int LDA  = (AST == 0) ? 1 : AST;                                  \
      const int BST  = static_cast<int>(B.stride(1));                         \
      const int LDB  = (BST == 0) ? 1 : BST;                                  \
      const int NRHS = static_cast<int>(B.extent(1));                         \
                                                                              \
      int info = 0;                                                           \
                                                                              \
      if (with_pivot) {                                                       \
        HostBlas<std::complex<double> >::getri(                                \
            N, NRHS, reinterpret_cast<std::complex<double>*>(A.data()), LDA,  \
            IPIV.data(), reinterpret_cast<std::complex<double>*>(B.data()),   \
            LDB, info);                                                       \
      }                                                                       \
      Kokkos::Profiling::popRegion();                                         \
    }                                                                         \
  };

#define KOKKOSBLAS_CGETRI_BLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)              \
  template <class ExecSpace>                                                  \
  struct GETRI<Kokkos::View<Kokkos::complex<float>**, LAYOUT,                  \
                           Kokkos::Device<ExecSpace, MEM_SPACE>,              \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,         \
              Kokkos::View<Kokkos::complex<float>**, LAYOUT,                  \
                           Kokkos::Device<ExecSpace, MEM_SPACE>,              \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,         \
              Kokkos::View<int*, LAYOUT,                                      \
                           Kokkos::Device<Kokkos::DefaultHostExecutionSpace,  \
                                          Kokkos::HostSpace>,                 \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,         \
              true, ETI_SPEC_AVAIL> {                                         \
    typedef Kokkos::complex<float> SCALAR;                                    \
    typedef Kokkos::View<SCALAR**, LAYOUT,                                    \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        AViewType;                                                            \
    typedef Kokkos::View<SCALAR**, LAYOUT,                                    \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        BViewType;                                                            \
    typedef Kokkos::View<                                                     \
        int*, LAYOUT,                                                         \
        Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace>, \
        Kokkos::MemoryTraits<Kokkos::Unmanaged> >                             \
        PViewType;                                                            \
                                                                              \
    static void getri(const AViewType& A, const BViewType& B,                  \
                     const PViewType& IPIV) {                                 \
      Kokkos::Profiling::pushRegion(                                          \
          "KokkosBlas::getri[TPL_BLAS,complex<float>]");                       \
      getri_print_specialization<AViewType, BViewType, PViewType>();           \
      const bool with_pivot =                                                 \
          !((IPIV.extent(0) == 0) && (IPIV.data() == nullptr));               \
                                                                              \
      const int N    = static_cast<int>(A.extent(1));                         \
      const int AST  = static_cast<int>(A.stride(1));                         \
      const int LDA  = (AST == 0) ? 1 : AST;                                  \
      const int BST  = static_cast<int>(B.stride(1));                         \
      const int LDB  = (BST == 0) ? 1 : BST;                                  \
      const int NRHS = static_cast<int>(B.extent(1));                         \
                                                                              \
      int info = 0;                                                           \
                                                                              \
      if (with_pivot) {                                                       \
        HostBlas<std::complex<float> >::getri(                                 \
            N, NRHS, reinterpret_cast<std::complex<float>*>(A.data()), LDA,   \
            IPIV.data(), reinterpret_cast<std::complex<float>*>(B.data()),    \
            LDB, info);                                                       \
      }                                                                       \
      Kokkos::Profiling::popRegion();                                         \
    }                                                                         \
  };

KOKKOSBLAS_DGETRI_BLAS(Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS_DGETRI_BLAS(Kokkos::LayoutLeft, Kokkos::HostSpace, false)

KOKKOSBLAS_SGETRI_BLAS(Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS_SGETRI_BLAS(Kokkos::LayoutLeft, Kokkos::HostSpace, false)

KOKKOSBLAS_ZGETRI_BLAS(Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS_ZGETRI_BLAS(Kokkos::LayoutLeft, Kokkos::HostSpace, false)

KOKKOSBLAS_CGETRI_BLAS(Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS_CGETRI_BLAS(Kokkos::LayoutLeft, Kokkos::HostSpace, false)

}  // namespace Impl
}  // namespace KokkosBlas
#endif  // KOKKOSKERNELS_ENABLE_TPL_BLAS

// MAGMA
#ifdef KOKKOSKERNELS_ENABLE_TPL_MAGMA
#include <KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS_DGETRI_MAGMA(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)             \
  template <class ExecSpace>                                                  \
  struct GETRI<                                                                \
      Kokkos::View<double**, LAYOUT, Kokkos::Device<ExecSpace, MEM_SPACE>,    \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<double**, LAYOUT, Kokkos::Device<ExecSpace, MEM_SPACE>,    \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<magma_int_t*, LAYOUT,                                      \
                   Kokkos::Device<Kokkos::DefaultHostExecutionSpace,          \
                                  Kokkos::HostSpace>,                         \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      true, ETI_SPEC_AVAIL> {                                                 \
    typedef double SCALAR;                                                    \
    typedef Kokkos::View<SCALAR**, LAYOUT,                                    \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        AViewType;                                                            \
    typedef Kokkos::View<SCALAR**, LAYOUT,                                    \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        BViewType;                                                            \
    typedef Kokkos::View<                                                     \
        magma_int_t*, LAYOUT,                                                 \
        Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace>, \
        Kokkos::MemoryTraits<Kokkos::Unmanaged> >                             \
        PViewType;                                                            \
                                                                              \
    static void getri(const AViewType& A, const BViewType& B,                  \
                     const PViewType& IPIV) {                                 \
      Kokkos::Profiling::pushRegion("KokkosBlas::getri[TPL_MAGMA,double]");    \
      getri_print_specialization<AViewType, BViewType, PViewType>();           \
      const bool with_pivot =                                                 \
          !((IPIV.extent(0) == 0) && (IPIV.data() == nullptr));               \
                                                                              \
      magma_int_t N    = static_cast<magma_int_t>(A.extent(1));               \
      magma_int_t AST  = static_cast<magma_int_t>(A.stride(1));               \
      magma_int_t LDA  = (AST == 0) ? 1 : AST;                                \
      magma_int_t BST  = static_cast<magma_int_t>(B.stride(1));               \
      magma_int_t LDB  = (BST == 0) ? 1 : BST;                                \
      magma_int_t NRHS = static_cast<magma_int_t>(B.extent(1));               \
                                                                              \
      KokkosBlas::Impl::MagmaSingleton& s =                                   \
          KokkosBlas::Impl::MagmaSingleton::singleton();                      \
      magma_int_t info = 0;                                                   \
                                                                              \
      if (with_pivot) {                                                       \
        magma_dgetri_gpu(N, NRHS, reinterpret_cast<magmaDouble_ptr>(A.data()), \
                        LDA, IPIV.data(),                                     \
                        reinterpret_cast<magmaDouble_ptr>(B.data()), LDB,     \
                        &info);                                               \
      } else {                                                                \
        magma_dgetri_nopiv_gpu(                                                \
            N, NRHS, reinterpret_cast<magmaDouble_ptr>(A.data()), LDA,        \
            reinterpret_cast<magmaDouble_ptr>(B.data()), LDB, &info);         \
      }                                                                       \
      Kokkos::Profiling::popRegion();                                         \
    }                                                                         \
  };

#define KOKKOSBLAS_SGETRI_MAGMA(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)             \
  template <class ExecSpace>                                                  \
  struct GETRI<                                                                \
      Kokkos::View<float**, LAYOUT, Kokkos::Device<ExecSpace, MEM_SPACE>,     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<float**, LAYOUT, Kokkos::Device<ExecSpace, MEM_SPACE>,     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<magma_int_t*, LAYOUT,                                      \
                   Kokkos::Device<Kokkos::DefaultHostExecutionSpace,          \
                                  Kokkos::HostSpace>,                         \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      true, ETI_SPEC_AVAIL> {                                                 \
    typedef float SCALAR;                                                     \
    typedef Kokkos::View<SCALAR**, LAYOUT,                                    \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        AViewType;                                                            \
    typedef Kokkos::View<SCALAR**, LAYOUT,                                    \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        BViewType;                                                            \
    typedef Kokkos::View<                                                     \
        magma_int_t*, LAYOUT,                                                 \
        Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace>, \
        Kokkos::MemoryTraits<Kokkos::Unmanaged> >                             \
        PViewType;                                                            \
                                                                              \
    static void getri(const AViewType& A, const BViewType& B,                  \
                     const PViewType& IPIV) {                                 \
      Kokkos::Profiling::pushRegion("KokkosBlas::getri[TPL_MAGMA,float]");     \
      getri_print_specialization<AViewType, BViewType, PViewType>();           \
      const bool with_pivot =                                                 \
          !((IPIV.extent(0) == 0) && (IPIV.data() == nullptr));               \
                                                                              \
      magma_int_t N    = static_cast<magma_int_t>(A.extent(1));               \
      magma_int_t AST  = static_cast<magma_int_t>(A.stride(1));               \
      magma_int_t LDA  = (AST == 0) ? 1 : AST;                                \
      magma_int_t BST  = static_cast<magma_int_t>(B.stride(1));               \
      magma_int_t LDB  = (BST == 0) ? 1 : BST;                                \
      magma_int_t NRHS = static_cast<magma_int_t>(B.extent(1));               \
                                                                              \
      KokkosBlas::Impl::MagmaSingleton& s =                                   \
          KokkosBlas::Impl::MagmaSingleton::singleton();                      \
      magma_int_t info = 0;                                                   \
                                                                              \
      if (with_pivot) {                                                       \
        magma_sgetri_gpu(N, NRHS, reinterpret_cast<magmaFloat_ptr>(A.data()),  \
                        LDA, IPIV.data(),                                     \
                        reinterpret_cast<magmaFloat_ptr>(B.data()), LDB,      \
                        &info);                                               \
      } else {                                                                \
        magma_sgetri_nopiv_gpu(                                                \
            N, NRHS, reinterpret_cast<magmaFloat_ptr>(A.data()), LDA,         \
            reinterpret_cast<magmaFloat_ptr>(B.data()), LDB, &info);          \
      }                                                                       \
      Kokkos::Profiling::popRegion();                                         \
    }                                                                         \
  };

#define KOKKOSBLAS_ZGETRI_MAGMA(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)             \
  template <class ExecSpace>                                                  \
  struct GETRI<Kokkos::View<Kokkos::complex<double>**, LAYOUT,                 \
                           Kokkos::Device<ExecSpace, MEM_SPACE>,              \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,         \
              Kokkos::View<Kokkos::complex<double>**, LAYOUT,                 \
                           Kokkos::Device<ExecSpace, MEM_SPACE>,              \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,         \
              Kokkos::View<magma_int_t*, LAYOUT,                              \
                           Kokkos::Device<Kokkos::DefaultHostExecutionSpace,  \
                                          Kokkos::HostSpace>,                 \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,         \
              true, ETI_SPEC_AVAIL> {                                         \
    typedef Kokkos::complex<double> SCALAR;                                   \
    typedef Kokkos::View<SCALAR**, LAYOUT,                                    \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        AViewType;                                                            \
    typedef Kokkos::View<SCALAR**, LAYOUT,                                    \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        BViewType;                                                            \
    typedef Kokkos::View<                                                     \
        magma_int_t*, LAYOUT,                                                 \
        Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace>, \
        Kokkos::MemoryTraits<Kokkos::Unmanaged> >                             \
        PViewType;                                                            \
                                                                              \
    static void getri(const AViewType& A, const BViewType& B,                  \
                     const PViewType& IPIV) {                                 \
      Kokkos::Profiling::pushRegion(                                          \
          "KokkosBlas::getri[TPL_MAGMA,complex<double>]");                     \
      getri_print_specialization<AViewType, BViewType, PViewType>();           \
      const bool with_pivot =                                                 \
          !((IPIV.extent(0) == 0) && (IPIV.data() == nullptr));               \
                                                                              \
      magma_int_t N    = static_cast<magma_int_t>(A.extent(1));               \
      magma_int_t AST  = static_cast<magma_int_t>(A.stride(1));               \
      magma_int_t LDA  = (AST == 0) ? 1 : AST;                                \
      magma_int_t BST  = static_cast<magma_int_t>(B.stride(1));               \
      magma_int_t LDB  = (BST == 0) ? 1 : BST;                                \
      magma_int_t NRHS = static_cast<magma_int_t>(B.extent(1));               \
                                                                              \
      KokkosBlas::Impl::MagmaSingleton& s =                                   \
          KokkosBlas::Impl::MagmaSingleton::singleton();                      \
      magma_int_t info = 0;                                                   \
                                                                              \
      if (with_pivot) {                                                       \
        magma_zgetri_gpu(                                                      \
            N, NRHS, reinterpret_cast<magmaDoubleComplex_ptr>(A.data()), LDA, \
            IPIV.data(), reinterpret_cast<magmaDoubleComplex_ptr>(B.data()),  \
            LDB, &info);                                                      \
      } else {                                                                \
        magma_zgetri_nopiv_gpu(                                                \
            N, NRHS, reinterpret_cast<magmaDoubleComplex_ptr>(A.data()), LDA, \
            reinterpret_cast<magmaDoubleComplex_ptr>(B.data()), LDB, &info);  \
      }                                                                       \
      Kokkos::Profiling::popRegion();                                         \
    }                                                                         \
  };

#define KOKKOSBLAS_CGETRI_MAGMA(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)             \
  template <class ExecSpace>                                                  \
  struct GETRI<Kokkos::View<Kokkos::complex<float>**, LAYOUT,                  \
                           Kokkos::Device<ExecSpace, MEM_SPACE>,              \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,         \
              Kokkos::View<Kokkos::complex<float>**, LAYOUT,                  \
                           Kokkos::Device<ExecSpace, MEM_SPACE>,              \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,         \
              Kokkos::View<magma_int_t*, LAYOUT,                              \
                           Kokkos::Device<Kokkos::DefaultHostExecutionSpace,  \
                                          Kokkos::HostSpace>,                 \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged> >,         \
              true, ETI_SPEC_AVAIL> {                                         \
    typedef Kokkos::complex<float> SCALAR;                                    \
    typedef Kokkos::View<SCALAR**, LAYOUT,                                    \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        AViewType;                                                            \
    typedef Kokkos::View<SCALAR**, LAYOUT,                                    \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        BViewType;                                                            \
    typedef Kokkos::View<                                                     \
        magma_int_t*, LAYOUT,                                                 \
        Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace>, \
        Kokkos::MemoryTraits<Kokkos::Unmanaged> >                             \
        PViewType;                                                            \
                                                                              \
    static void getri(const AViewType& A, const BViewType& B,                  \
                     const PViewType& IPIV) {                                 \
      Kokkos::Profiling::pushRegion(                                          \
          "KokkosBlas::getri[TPL_MAGMA,complex<float>]");                      \
      getri_print_specialization<AViewType, BViewType, PViewType>();           \
      const bool with_pivot =                                                 \
          !((IPIV.extent(0) == 0) && (IPIV.data() == nullptr));               \
                                                                              \
      magma_int_t N    = static_cast<magma_int_t>(A.extent(1));               \
      magma_int_t AST  = static_cast<magma_int_t>(A.stride(1));               \
      magma_int_t LDA  = (AST == 0) ? 1 : AST;                                \
      magma_int_t BST  = static_cast<magma_int_t>(B.stride(1));               \
      magma_int_t LDB  = (BST == 0) ? 1 : BST;                                \
      magma_int_t NRHS = static_cast<magma_int_t>(B.extent(1));               \
                                                                              \
      KokkosBlas::Impl::MagmaSingleton& s =                                   \
          KokkosBlas::Impl::MagmaSingleton::singleton();                      \
      magma_int_t info = 0;                                                   \
                                                                              \
      if (with_pivot) {                                                       \
        magma_cgetri_gpu(                                                      \
            N, NRHS, reinterpret_cast<magmaFloatComplex_ptr>(A.data()), LDA,  \
            IPIV.data(), reinterpret_cast<magmaFloatComplex_ptr>(B.data()),   \
            LDB, &info);                                                      \
      } else {                                                                \
        magma_cgetri_nopiv_gpu(                                                \
            N, NRHS, reinterpret_cast<magmaFloatComplex_ptr>(A.data()), LDA,  \
            reinterpret_cast<magmaFloatComplex_ptr>(B.data()), LDB, &info);   \
      }                                                                       \
      Kokkos::Profiling::popRegion();                                         \
    }                                                                         \
  };

KOKKOSBLAS_DGETRI_MAGMA(Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS_DGETRI_MAGMA(Kokkos::LayoutLeft, Kokkos::CudaSpace, false)

KOKKOSBLAS_SGETRI_MAGMA(Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS_SGETRI_MAGMA(Kokkos::LayoutLeft, Kokkos::CudaSpace, false)

KOKKOSBLAS_ZGETRI_MAGMA(Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS_ZGETRI_MAGMA(Kokkos::LayoutLeft, Kokkos::CudaSpace, false)

KOKKOSBLAS_CGETRI_MAGMA(Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS_CGETRI_MAGMA(Kokkos::LayoutLeft, Kokkos::CudaSpace, false)

}  // namespace Impl
}  // namespace KokkosBlas
#endif  // KOKKOSKERNELS_ENABLE_TPL_MAGMA

#endif
