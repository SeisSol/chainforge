from .default_kernel import DefaultKernelBuilder
from .single_warp_kernel import SingleWarpKernelBuilder
from .kernel_types import KernelType
from chainforge.common import Context
from chainforge.backend.scopes import Scopes


def kernel_factory(context: Context,
                   scopes: Scopes,
                   gemm_list,
                   kernel_type):
  if kernel_type == KernelType.AUTO:
    return DefaultKernelBuilder(context, scopes, gemm_list)
  elif kernel_type == KernelType.DEFAULT:
    return DefaultKernelBuilder(context, scopes, gemm_list)
  elif kernel_type == KernelType.SINGLE_WARP:
    return SingleWarpKernelBuilder(context, scopes, gemm_list)
  else:
    raise RuntimeError('unknown kernel type')
