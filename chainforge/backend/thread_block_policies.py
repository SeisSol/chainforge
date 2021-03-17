from chainforge.common import VM
from math import floor
from abc import ABC, abstractmethod


class AbstractThreadBlockPolicy(ABC):
  def __init__(self, vm: VM, mem_per_mult: int, num_threads: int):
    self._vm: VM = vm
    self._mem_per_mult: int = mem_per_mult
    self._num_threads: int = num_threads
    self._max_blocks = self._vm.hw_descr.max_block_per_sm
    self._max_allowed_mem = self._vm.hw_descr.max_local_mem_size_per_block

  @abstractmethod
  def get_num_mults_per_block(self):
    pass


class SimpleThreadBlockPolicy(AbstractThreadBlockPolicy):
  def __init__(self, vm, mem_size_per_mult, num_threads):
    super().__init__(vm, mem_size_per_mult, num_threads)

  def get_num_mults_per_block(self):
    if self._num_threads <= 32:
      return 2
    else:
      return 1
