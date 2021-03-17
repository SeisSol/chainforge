from .abstract_instruction import AbstractInstruction
from chainforge.common.vm import VM
from chainforge.common.basic_types import FloatingPointType
from chainforge.backend.symbol import Symbol, SymbolType
from chainforge.backend.writer import Writer
from chainforge.backend.exceptions import InternalError


class SyncThreads(AbstractInstruction):
  def __init__(self, vm: VM, num_threads_per_mult):
    super().__init__(vm)
    self._num_threads = num_threads_per_mult
    self._is_ready = True

  def gen_code(self, writer: Writer):
    writer(f'{self.__str__()}')

  def __str__(self) -> str:
    if self._num_threads > self._vm.hw_descr.vec_unit_length:
      return f'{self._vm.lexic.sync_block_threads};'
    else:
      return f'{self._vm.lexic.sync_warp_threads};'

  def gen_mask_threads(self, num_threads) -> str:
    return ''
