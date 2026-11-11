//! Basic module traits defining the forward pass.
use crate::Tensor;

/// The simplest module trait, defining a forward function.
pub trait Module: std::fmt::Debug + Send {
    fn forward(&self, xs: &Tensor) -> Tensor;
}

/// Module trait with an additional train parameter.
///
/// The train parameter is commonly used to have different behavior between training
/// and evaluation, e.g. when using dropout or batch-normalization.
pub trait ModuleT: std::fmt::Debug + Send {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor;
}

impl<T> ModuleT for T
where
    T: Module,
{
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        self.forward(xs)
    }
}

impl Tensor {
    pub fn apply<M: Module>(&self, m: &M) -> Tensor {
        m.forward(self)
    }

    pub fn apply_t<M: ModuleT>(&self, m: &M, train: bool) -> Tensor {
        m.forward_t(self, train)
    }

    pub fn apply_opt<M: Module>(&self, m: &Option<M>) -> Tensor {
        match m {
            Some(m) => m.forward(self),
            None => self.shallow_clone(),
        }
    }

    pub fn apply_opt_t<M: ModuleT>(&self, m: &Option<M>, train: bool) -> Tensor {
        match m {
            Some(m) => m.forward_t(self, train),
            None => self.shallow_clone(),
        }
    }
}
