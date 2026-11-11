use crate::{ffi::NativeCall, sys};

/// Number of physical CPU cores estimated by stable-diffusion.cpp.
#[must_use]
pub fn physical_core_count() -> i32 {
    let _call = NativeCall::enter();
    unsafe { sys::sd_get_num_physical_cores() }
}
