from chainforge.common.vm import VM
from chainforge.backend.writer import Writer
from abc import ABC, abstractmethod
from typing import Union


class AbstractInstruction(ABC):
  def __init__(self, vm: VM):
    self._vm: VM = vm
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
    return f'if ({self._vm.lexic.threadIdx_x} < {num_threads})'


class AbstractShrMemWrite(AbstractInstruction):
  def __init__(self, vm: VM):
    super().__init__(vm)
    self._shm_volume: int = 0
    self._shr_mem_offset: Union[int, None] = 0

  def compute_shared_mem_size(self) -> int:
    return self._shm_volume

  def set_shr_mem_offset(self, offset: int) -> None:
    self._shr_mem_offset = offset
    self._is_ready = True
