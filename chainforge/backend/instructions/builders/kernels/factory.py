from .default_kernel import DefaultKernelBuilder
from .single_warp_kernel import SingleWarpKernelBuilder
from chainforge.common import Context
from chainforge.backend.scopes import Scopes
from enum import Enum


class KernelType(Enum):
  AUTO = 0
  DEFAULT = 1
  MIN_THREADS = 2


def kernel_factory(context: Context,
                   scopes: Scopes,
                   gemm_list,
                   kernel_type):
  if kernel_type == KernelType.AUTO:
    return DefaultKernelBuilder(context, scopes, gemm_list)
  elif kernel_type == KernelType.DEFAULT:
    return DefaultKernelBuilder(context, scopes, gemm_list)
  elif kernel_type == KernelType.MIN_THREADS:
    return SingleWarpKernelBuilder(context, scopes, gemm_list)
  else:
    raise RuntimeError('unknown kernel type')
