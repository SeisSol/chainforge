from abc import ABC, abstractmethod
from chainforge.common import Context


class AbstractThreadBlockPolicy(ABC):
  def __init__(self, context: Context, mem_per_mult: int, num_threads: int):
    self._context: Context = context
    self._mem_per_mult: int = mem_per_mult
    self._num_threads: int = num_threads

    vm = self._context.get_vm()
    self._max_blocks = vm.hw_descr.max_block_per_sm
    self._max_allowed_mem = vm.hw_descr.max_local_mem_size_per_block

  @abstractmethod
  def get_num_mults_per_block(self):
    pass


class SimpleThreadBlockPolicy(AbstractThreadBlockPolicy):
  def __init__(self, context, mem_size_per_mult, num_threads):
    super().__init__(context, mem_size_per_mult, num_threads)

  def get_num_mults_per_block(self):
    if self._num_threads <= 32:
      return 2
    else:
      return 1
