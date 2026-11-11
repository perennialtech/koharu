use std::{mem::forget, path::Path};

use anyhow::Result;

pub(super) fn load(path: impl AsRef<Path>, global: bool) -> Result<()> {
    let path = path.as_ref();
    let path = dunce::canonicalize(path)?;
    let library = unsafe { open(&path, global) }?;
    forget(library);
    Ok(())
}

unsafe fn open(path: &Path, global: bool) -> Result<libloading::Library, libloading::Error> {
    use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_LAZY, RTLD_LOCAL};
    let visibility = if global { RTLD_GLOBAL } else { RTLD_LOCAL };
    unsafe { Library::open(Some(path), RTLD_LAZY | visibility).map(Into::into) }
}
