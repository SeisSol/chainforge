from typing import List, Dict, Set
from chainforge.common import Context
from chainforge.backend.symbol import Symbol
from chainforge.backend.instructions import AbstractInstruction
from chainforge.backend.data_types import ShrMemObject
from .liveness import LivenessAnalysis
from .mem_region_allocation import MemoryRegionAllocation, Region
from .shr_mem_analyzer import ShrMemOpt
from .sync_threads import SyncThreadsOpt
from .remove_redundancy import RemoveRedundancyOpt


class OptimizationStage:
  def __init__(self,
               context: Context,
               shr_mem: ShrMemObject,
               instructions: List[AbstractInstruction],
               num_threads: int):
    self._context = context
    self._shr_mem: ShrMemObject = shr_mem
    self._instrs: List[AbstractInstruction] = instructions
    self._num_instrs: int = len(instructions)
    self._user_options = context.get_user_options()
    self._num_threads = num_threads

  def optimize(self):
    opt = LivenessAnalysis(self._context, self._instrs)
    opt.apply()
    live_map: Dict[int, Set[Symbol]] = opt.get_live_map()

    opt = MemoryRegionAllocation(self._context, live_map)
    opt.apply()
    regions: List[Region] = opt.get_regions()

    opt = ShrMemOpt(context=self._context,
                    shr_mem_obj=self._shr_mem,
                    regions=regions,
                    live_map=live_map)
    opt.apply()

    if self._user_options.enable_sync_threads_opt:
      opt = SyncThreadsOpt(self._context, self._instrs, regions, self._num_threads)
      opt.apply()
      self._instrs = opt.get_instructions()

    opt = RemoveRedundancyOpt(self._context, self._instrs)
    opt.apply()
    self._instrs = opt.get_instructions()

  def get_instructions(self):
    return self._instrs
