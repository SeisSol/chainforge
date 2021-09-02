from .abstract import AbstractTransformer, Context, AbstractInstruction
from chainforge.backend.instructions import Gemm, SyncThreads, AbstractShrMemWrite
from chainforge.backend.symbol import SymbolType
from .mem_region_allocation import Region
from typing import List


class SyncThreadsOpt(AbstractTransformer):
  def __init__(self,
               context: Context,
               instructions: List[AbstractInstruction],
               regions: List[Region],
               num_threads: int):

    super(SyncThreadsOpt, self).__init__(context, instructions)
    self._regions = regions
    self._num_threads = num_threads

  def apply(self) -> None:
    self._remove_previous_sync_instructions()
    self._insert_sync_before_use()
    self._insert_sync_after_use()

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
      self._instrs.insert(index, SyncThreads(self._context, self._num_threads))

  def _get_region_id(self, symbol):
    for region_id, region in enumerate(self._regions):
      if symbol in region:
        return region_id

  def _remove_previous_sync_instructions(self):
    self._instrs = [item for item in self._instrs if not isinstance(item, SyncThreads)]
