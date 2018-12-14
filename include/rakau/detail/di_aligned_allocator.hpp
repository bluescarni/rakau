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
#include <cstdlib>
#include <new>
#include <type_traits>
#include <utility>

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
        const auto size = n * sizeof(T);
        void *retval;
        if (Alignment == 0u) {
            retval = std::malloc(size);
        } else {
#if defined(__apple_build_version__)
            // NOTE: std::aligned_alloc() is apparently not (yet) available on OSX.
            // Use posix_memalign() instead.
            // NOTE: posix_memalign() returns 0 on success. In case of errors,
            // we will set retval to nullptr to signal that the allocation failed
            // (so that we can handle the allocation failure in the same codepath
            // as aligned_alloc()).
            if (::posix_memalign(&retval, Alignment, size)) {
                retval = nullptr;
            }
#else
            // For use in std::aligned_alloc, the size must be a multiple of the alignment.
            // http://en.cppreference.com/w/cpp/memory/c/aligned_alloc
            // A null pointer will be returned if invalid Alignment and/or size are supplied,
            // or if the allocation fails.
            // NOTE: some early versions of GCC put aligned_alloc in the root namespace rather
            // than std, so let's try to workaround.
            using namespace std;
            retval = aligned_alloc(Alignment, size);
#endif
        }
        if (!retval) {
            // Throw on failure.
            throw std::bad_alloc{};
        }
        return static_cast<T *>(retval);
    }
    // Deallocation.
    void deallocate(T *ptr, size_type) const
    {
        std::free(ptr);
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
    template <typename U>
    void construct(U *p) const
    {
        // When no construction arguments are supplied, do default
        // initialisation rather than value initialisation.
        ::new (static_cast<void *>(p)) U;
    }
    template <typename U, typename... Args>
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
