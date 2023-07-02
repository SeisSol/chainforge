from abc import ABC, abstractmethod
from typing import Union
from chainforge.common import Context, VM
from chainforge.backend.writer import Writer


class AbstractInstruction(ABC):
  def __init__(self, context: Context):
    if not isinstance(context, Context):
      raise RuntimeError(f'received wrong type, expected Context, given {type(context)}')

    self._context = context
    self._vm: VM = context.get_vm()
    self._fp_as_str = context.fp_as_str()
    self._is_ready = False

  @abstractmethod
  def gen_code(self, writer: Writer) -> None:
    return None

  def is_ready(self) -> bool:
    return self._is_ready

  @abstractmethod
  def __str__(self) -> str:
    pass

  def gen_mask_threads(self, num_threads) -> str:
    return f'if ({self._vm.lexic.thread_idx_x} < {num_threads})'

  def gen_range_mask_threads(self, begin, end) -> str:
    assert begin < end
    tid = self._vm.lexic.thread_idx_x
    return f'if (({tid} >= {begin}) && ({tid} < {end}))'


class AbstractShrMemWrite(AbstractInstruction):
  def __init__(self, context: Context):
    super().__init__(context)
    self._shm_volume: int = 0
    self._shr_mem_offset: Union[int, None] = 0

  def compute_shared_mem_size(self) -> int:
    user_options = self._context.get_user_options()

    if user_options.align_shr_mem:
      size = self._context.align(self._shm_volume)
    else:
      size = self._shm_volume
    return size

  def set_shr_mem_offset(self, offset: int) -> None:
    self._shr_mem_offset = offset
    self._is_ready = True

  @abstractmethod
  def get_dest(self):
    pass
