from .liveness import LivenessAnalysis
from .mem_region_allocation import MemoryRegionAllocation, Region
from .shr_mem_opt import ShrMemOpt
from .sync_threads_opt import SyncThreadsOpt
from chainforge.backend.symbol import Symbol
from chainforge.backend.instructions import AbstractInstruction
from chainforge.backend.data_types import ShrMemObject
from typing import List, Dict, Set


class Optimizer:
  def __init__(self, shr_mem: ShrMemObject, instructions: List[AbstractInstruction]):
    self._shr_mem: ShrMemObject = shr_mem
    self._instrs: List[AbstractInstruction] = instructions
    self._num_instrs: int = len(instructions)

  def optimize(self):
    live = LivenessAnalysis(self._instrs)
    live_map: Dict[int, Set[Symbol]] = live.apply()

    allocator = MemoryRegionAllocation(live_map)
    regions: List[Region] = allocator.allocate_regions()

    shr_opt = ShrMemOpt(shr_mem_obj=self._shr_mem,
                        regions=regions,
                        live_map=live_map)
    shr_opt.apply()

    sync_threads_opt = SyncThreadsOpt(self._instrs, regions)
    sync_threads_opt.remove_redundant_syncs()
