/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOSBLAS1_TEAM_SCAL_HPP_
#define KOKKOSBLAS1_TEAM_SCAL_HPP_

#include <KokkosBlas1_team_scal_impl.hpp>

// TODO: deprecate/remove ?
#include <KokkosBlas1_team_scal_spec.hpp>

namespace KokkosBlas {

///
/// Team Scale
///

template <typename MemberType>
struct TeamScale {
  template <typename ScalarType, typename AViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType& member,
                                           const ScalarType alpha,
                                           const AViewType& A) {
    return Impl::TeamScaleInternal::invoke(member, A.extent(0), A.extent(1),
                                           alpha, A.data(), A.stride_0(),
                                           A.stride_1());
  }
};

///
/// TeamVector Scale
///

template <typename MemberType>
struct TeamVectorScale {
  template <typename ScalarType, typename AViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType& member,
                                           const ScalarType alpha,
                                           const AViewType& A) {
    return Impl::TeamVectorScaleInternal::invoke(member, A.extent(0),
                                                 A.extent(1), alpha, A.data(),
                                                 A.stride_0(), A.stride_1());
  }
};

// TODO: deprecate/remove ?
namespace Experimental {

template <class TeamType, class RVector, class XVector>
void KOKKOS_INLINE_FUNCTION
scal(const TeamType& team, const RVector& r,
     const typename XVector::non_const_value_type& a, const XVector& x) {
  return Impl::TeamScal<TeamType, RVector, XVector>::team_scal(team, r, a, x);
}

}  // namespace Experimental
}  // namespace KokkosBlas

#endif
