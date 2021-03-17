from chainforge.backend.instructions import AbstractInstruction
from chainforge.backend.instructions import SyncThreads, ClearRegisters
from chainforge.backend.instructions import StoreRegToGlb

from typing import List


class SyncThreadsOpt:
  def __init__(self, instructions: List[AbstractInstruction]):
    self._instrs = instructions

  def remove_redundant_syncs(self):
    self._remove_bottom_instrs()
    pass

  def _remove_bottom_instrs(self):
    num_remove_instrs = 0
    for reversed_index, instr in enumerate(reversed(self._instrs)):
      num_remove_instrs += 1
      if isinstance(instr, StoreRegToGlb):
        break

    for index in range(num_remove_instrs - 1):
      self._instrs.pop(-1)
