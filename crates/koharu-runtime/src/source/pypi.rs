pub(crate) struct Wheel {
    pub(crate) project: &'static str,
    pub(crate) version: &'static str,
    pub(crate) platform: &'static str,
}

impl Wheel {
    pub(crate) fn directory(&self) -> String {
        format!("{}--{}", self.project, self.version)
    }

    /// PyPI redirects this filename-based URL to the wheel's storage location.
    pub(crate) fn url(&self) -> String {
        let Self {
            project,
            version,
            platform,
        } = self;
        let distribution = project.replace('-', "_");
        let first = &project[..1];
        format!(
            "https://files.pythonhosted.org/packages/py3/{first}/{project}/{distribution}-{version}-py3-none-{platform}.whl"
        )
    }
}
