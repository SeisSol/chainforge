from abc import ABC, abstractmethod
from chainforge.common import Context
from chainforge.backend.instructions.builders.kernels import KernelType


class AbstractThreadBlockPolicy(ABC):
  def __init__(self,
               context: Context,
               kernel_type: KernelType,
               mem_per_mult: int,
               num_threads: int):
    self._context: Context = context
    self._kernel_type = kernel_type
    self._mem_per_mult: int = mem_per_mult
    self._num_threads: int = num_threads

    vm = self._context.get_vm()
    self._max_blocks = vm.hw_descr.max_block_per_sm
    self._max_allowed_mem = vm.hw_descr.max_local_mem_size_per_block

  @abstractmethod
  def get_num_mults_per_block(self):
    pass


class SimpleThreadBlockPolicy(AbstractThreadBlockPolicy):
  def __init__(self,
               context,
               kernel_type: KernelType,
               mem_size_per_mult,
               num_threads):
    super().__init__(context, kernel_type, mem_size_per_mult, num_threads)

  def get_num_mults_per_block(self):
    num_mults = 1
    if self._kernel_type == KernelType.DEFAULT:
      if self._num_threads <= 32:
        num_mults = 2
    return num_mults
