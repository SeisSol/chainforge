from chainforge.backend.instructions import AbstractInstruction
from chainforge.backend.instructions import SyncThreads, ClearRegisters
from chainforge.backend.instructions import StoreRegToGlb
from chainforge.backend.instructions import Gemm, StoreRegToShr, SyncThreads, AbstractShrMemWrite
from chainforge.backend.symbol import SymbolType
from .mem_region_allocation import Region
from typing import List


class SyncThreadsOpt:
  def __init__(self, instructions: List[AbstractInstruction], regions: List[Region]):
    self._instrs = instructions
    self._regions = regions
    self._vm = self._instrs[0]._vm

  def remove_redundant_syncs(self):
    self._remove_bottom_instrs()
    self._insert_sync_before_use()
    self._insert_sync_after_use()
    self._print_instr()

  def _print_instr(self):
    for index, instr in enumerate(self._instrs):
      print(f'{index}:   {instr}')

  def _insert_sync_before_use(self):
    selected = []
    writes = []
    for index, instr in enumerate(self._instrs):
      if isinstance(instr, AbstractShrMemWrite):
        writes.append(instr.get_dest())

      if isinstance(instr, Gemm):
        if instr.get_op1() in writes or instr.get_op2() in writes:
          selected.append(instr)
          writes = []

    self._insert_sync_instrs(selected)

  def _insert_sync_after_use(self):
    selected = []
    flags = [False] * len(self._regions)
    for index, instr in enumerate(self._instrs):
      if isinstance(instr, Gemm):
        for src in [instr.get_op1(), instr.get_op2()]:
          if src.stype == SymbolType.SharedMem:
            flags[self._get_region_id(src)] = True

      if isinstance(instr, SyncThreads):
        flags = [False] * len(self._regions)

      if isinstance(instr, AbstractShrMemWrite):
        dest = instr.get_dest()
        if flags[self._get_region_id(dest)]:
          selected.append(instr)
          flags = [False] * len(self._regions)

    self._insert_sync_instrs(selected)

  def _insert_sync_instrs(self, selected):
    for instr in selected:
      index = self._instrs.index(instr)
      self._instrs.insert(index, SyncThreads(self._vm, 64))

  def _get_region_id(self, symbol):
    for region_id, region in enumerate(self._regions):
      if symbol in region:
        return region_id

  def _remove_bottom_instrs(self):
    num_remove_instrs = 0
    for reversed_index, instr in enumerate(reversed(self._instrs)):
      num_remove_instrs += 1
      if isinstance(instr, StoreRegToGlb):
        break

    for index in range(num_remove_instrs - 1):
      self._instrs.pop(-1)
