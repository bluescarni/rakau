// Copyright 2018 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the rakau library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef RAKAU_DETAIL_DI_ALIGNED_ALLOCATOR_HPP
#define RAKAU_DETAIL_DI_ALIGNED_ALLOCATOR_HPP

#include <cstddef>
#include <new>
#include <type_traits>
#include <utility>

#include <rakau/config.hpp>

#if defined(RAKAU_WITH_CUDA)

#include <cassert>
#include <limits>
#include <memory>

#include <cuda_runtime_api.h>

#else

#include <cstdlib>

#endif

namespace rakau
{

inline namespace detail
{

// A trivial allocator that supports custom alignment and does
// default-initialisation instead of value-initialisation.
template <typename T, std::size_t Alignment = 0>
struct di_aligned_allocator {
    // Alignment must be either zero or:
    // - not less than the natural alignment of T,
    // - a power of 2.
    static_assert(
        !Alignment || (Alignment >= alignof(T) && (Alignment & (Alignment - 1u)) == 0u),
        "Invalid alignment value: the alignment must be a power of 2 and not less than the natural alignment of T.");
    // value_type must always be present.
    using value_type = T;
    // Make sure the size_type is consistent with the size type returned
    // by malloc() and friends.
    using size_type = std::size_t;
    // NOTE: rebind is needed because we have a non-type template parameter, which
    // prevents the automatic implementation of rebind from working.
    // http://en.cppreference.com/w/cpp/concept/Allocator#cite_note-1
    template <typename U>
    struct rebind {
        using other = di_aligned_allocator<U, Alignment>;
    };
    // Match the behaviour of std::allocator on these type traits.
    // https://en.cppreference.com/w/cpp/memory/allocator
    using is_always_equal = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    // Allocation.
    T *allocate(size_type n) const
    {
        // Total size in bytes. This is prevented from being too large
        // by the default implementation of max_size().
        auto size = n * sizeof(T);
        void *retval;
#if defined(RAKAU_WITH_CUDA)
        // The alignment value must fit in 1 byte.
        static_assert(Alignment <= std::numeric_limits<unsigned char>::max());
        // ptrdiff_t must be able to represent the alignment value.
        static_assert(Alignment
                      <= static_cast<std::make_unsigned_t<std::ptrdiff_t>>(std::numeric_limits<std::ptrdiff_t>::max()));

        // Guard against overflow.
        if (size > std::numeric_limits<std::size_t>::max() - Alignment) {
            throw std::bad_alloc{};
        }
        // Add the alignment.
        size += Alignment;

        // Try to allocate.
        if (::cudaMallocManaged(&retval, size) != ::cudaSuccess) {
            throw std::bad_alloc{};
        }

        if (Alignment != 0u) {
            // With a nonzero alignment, we need to manipulate the returned value.
            // Store the original pointer.
            const auto orig_retval = retval;

            // Now try to align.
            if (std::align(Alignment, size - Alignment, retval, size) == nullptr) {
                // Alignment failed. Free and throw.
                // NOTE: if std::align fails, it does not modify retval (which will
                // still then be the original pointer).
                ::cudaFree(retval);
                throw std::bad_alloc{};
            }

            // Alignment was successful, now retval points to the aligned address.
            // If the aligned address is the original one, bump it up by Alignment.
            if (retval == orig_retval) {
                retval = static_cast<void *>(reinterpret_cast<unsigned char *>(retval) + Alignment);
            }
            // Compute the distance in bytes between retval and the original pointer.
            // NOTE: this cannot be larger than Alignment, because otherwise std::align would've failed.
            const auto ptr_diff
                = reinterpret_cast<unsigned char *>(retval) - reinterpret_cast<unsigned char *>(orig_retval);
            // Store the distance in the byte immediately preceding retval.
            *(reinterpret_cast<unsigned char *>(retval) - 1) = static_cast<unsigned char>(ptr_diff);
        }
#else
        if (Alignment == 0u) {
            retval = std::malloc(size);
        } else {
            // For use in std::aligned_alloc, the size must be a multiple of the alignment.
            // http://en.cppreference.com/w/cpp/memory/c/aligned_alloc
            // A null pointer will be returned if invalid Alignment and/or size are supplied,
            // or if the allocation fails.
            // NOTE: some early versions of GCC put aligned_alloc in the root namespace rather
            // than std, so let's try to workaround.
            using namespace std;
            retval = aligned_alloc(Alignment, size);
        }
        if (!retval) {
            // Throw on failure.
            throw std::bad_alloc{};
        }
#endif
        return static_cast<T *>(retval);
    }
    // Deallocation.
    void deallocate(T *ptr, size_type) const
    {
#if defined(RAKAU_WITH_CUDA)
        if (Alignment == 0u) {
            ::cudaFree(ptr);
        } else {
            const auto delta = *(reinterpret_cast<unsigned char *>(ptr) - 1);
            ::cudaFree(static_cast<void *>(reinterpret_cast<unsigned char *>(ptr) - delta));
        }
#else
        std::free(ptr);
#endif
    }
    // Trivial comparison operators.
    friend bool operator==(const di_aligned_allocator &, const di_aligned_allocator &)
    {
        return true;
    }
    friend bool operator!=(const di_aligned_allocator &, const di_aligned_allocator &)
    {
        return false;
    }
    // The construction function.
    template <typename U, typename... Args, std::enable_if_t<sizeof...(Args) == 0u, int> = 0>
    void construct(U *p, Args &&... args) const
    {
        // When no construction arguments are supplied, do default
        // initialisation rather than value initialisation.
        ::new (static_cast<void *>(p)) U;
    }
    template <typename U, typename... Args, std::enable_if_t<sizeof...(Args) != 0u, int> = 0>
    void construct(U *p, Args &&... args) const
    {
        // This is the standard std::allocator implementation.
        // http://en.cppreference.com/w/cpp/memory/allocator/construct
        ::new (static_cast<void *>(p)) U(std::forward<Args>(args)...);
    }
};

} // namespace detail

} // namespace rakau

#endif
