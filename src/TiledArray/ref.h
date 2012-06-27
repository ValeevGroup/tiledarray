#ifndef TILEDARRAY_REF_H__INCLUDED
#define TILEDARRAY_REF_H__INCLUDED

#include <TiledArray/error.h>
#include <world/type_traits.h>

//  ref.hpp - ref/cref, useful helper functions
//
//  Copyright (C) 1999, 2000 Jaakko Jarvi (jaakko.jarvi@cs.utu.fi)
//  Copyright (C) 2001, 2002 Peter Dimov
//  Copyright (C) 2002 David Abrahams
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
//  See http://www.boost.org/libs/bind/ref.html for documentation.
//
//  Ported by JAC into madness 6/27/2012

namespace TiledArray {
  namespace detail {

    /// Reference wrapper class

    /// Used to hold references for task functions where a copy would other wise
    /// be used. This wrapper object is default constructable, copy constructable,
    /// assignable, and serializable, while a normal reference would not be.
    /// \tparam T The reference type
    template<class T>
    class ReferenceWrapper {
    private:
        T* t_;
    public:
        typedef T type; ///< The reference type

        /// Default constructor

        /// The reference references nothing
        ReferenceWrapper() : t_(NULL) { }

        /// Constructor

        /// \param t The object to reference
        explicit ReferenceWrapper(T& t) : t_(&t) { }

        // Compiler generated copy constructor and assignment operator are OK here

        /// Reference accessor

        /// \return A reference to the referenced object
        T& get() const {
          TA_ASSERT(t_);
          return *t_;
        }

        /// Type conversion operator

        /// This has the same effect as \c get() .
        /// \return A reference to the reference object
        operator T& () const { return get(); }

        /// Obect pointer accessor

        /// \return A pointer to the referenced object
        T* get_pointer() const { return t_; }

        /// Serialization

        /// This function is here for compatibility with task functions. Since
        /// serializing a reference to a local object is inheirently wrong, this
        /// function simply throws an exception.
        /// \throw TiledArray::Exception Always
        template <typename Archive>
        void serialize(const Archive&) { TA_EXCEPTION("ReferenceWrapper serialization not supported."); }

    }; // class ReferenceWrapper

    /// Reference wrapper factory function

    /// \tparam T The reference type (may be const or non-const)
    /// \param t The object to be wrapped
    /// \return A reference wrapper object
    template<class T>
    inline ReferenceWrapper<T> const ref(T& t) {  return ReferenceWrapper<T>(t); }

    /// Constant reference wrapper factory function

    /// \tparam T The reference type (without const)
    /// \param t The object to be wrapped
    /// \return Constant reference wrapper object
    template<class T>
    inline ReferenceWrapper<T const> const cref(const T& t) { return ReferenceWrapper<const T>(t); }


    /// Type trait for reference wrapper

    /// \tparam T The test type
    template<typename T>
    class is_reference_wrapper : public std::false_type { };

    /// \c ReferenceWrapper type trait accessor
    template<typename T>
    class UnwrapReference {
    public:
      typedef T type; ///< The reference type
    }; // class UnwrapReference

    template<typename T>
    class is_reference_wrapper<ReferenceWrapper<T> > : public std::true_type { };

    template<typename T>
    class is_reference_wrapper<ReferenceWrapper<T> const> : public std::true_type { };

    template<typename T>
    class is_reference_wrapper<ReferenceWrapper<T> volatile> : public std::true_type { };

    template<typename T>
    class is_reference_wrapper<ReferenceWrapper<T> const volatile> : public std::true_type { };

    template<typename T>
    class UnwrapReference<ReferenceWrapper<T> > {
     public:
        typedef T type;
    }; // class UnwrapReference<ReferenceWrapper<T> >

    template<typename T>
    class UnwrapReference<ReferenceWrapper<T> const> {
     public:
        typedef T type;
    }; // class UnwrapReference<ReferenceWrapper<T> const>

    template<typename T>
    class UnwrapReference<ReferenceWrapper<T> volatile> {
     public:
        typedef T type;
    }; // class UnwrapReference<ReferenceWrapper<T> volatile>

    template<typename T>
    class UnwrapReference<ReferenceWrapper<T> const volatile> {
     public:
        typedef T type;
    }; // class UnwrapReference<ReferenceWrapper<T> const volatile>


    /// Function for retreaving the referenced object

    /// \tparam The reference type
    /// \param t The reference being unwrapped
    /// \return A reference to the original objects
    template <class T>
    inline typename UnwrapReference<T>::type& unwrap_ref(T& t) { return t; }

    /// Function for retreaving a pointer to the referenced object

    /// \tparam The reference type
    /// \param t The ReferenceWrapper object
    /// \return A reference to the original objects
    template<class T>
    inline T* get_pointer(const ReferenceWrapper<T>& r ) { return r.get_pointer(); }


  }  // namespace detail
}  // namespace TiledArray

#endif // TILEDARRAY_REF_H__INCLUDED
