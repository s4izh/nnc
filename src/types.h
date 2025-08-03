#ifndef __TYPES_H__
#define __TYPES_H__

#include <stdint.h>
#include <stdbool.h>

// check if the compiler supports 128-bit integers
#if defined(__SIZEOF_INT128__)
typedef __int128 i128;
typedef unsigned __int128 u128;
#else
#error "128-bit integers are not supported by this compiler"
#endif

// signed integer types
typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

// unsigned integer types
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

// floating point types
typedef float f32;
typedef double f64;

#endif // __TYPES_H__
