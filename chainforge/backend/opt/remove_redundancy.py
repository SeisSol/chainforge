from .abstract import AbstractTransformer, Context, AbstractInstruction
from chainforge.backend.instructions import StoreRegToGlb
from typing import List


class RemoveRedundancyOpt(AbstractTransformer):
  def __init__(self,
               context: Context,
               instructions: List[AbstractInstruction]):
    super(RemoveRedundancyOpt, self).__init__(context, instructions)

  def apply(self) -> None:
    self._remove_bottom_instrs()

  def _remove_bottom_instrs(self):
    """
    The last instruction - i.e., clean register - produced by GemmBuilder is redundant and
    can be removed
    """
    num_remove_instrs = 0
    for reversed_index, instr in enumerate(reversed(self._instrs)):
      num_remove_instrs += 1
      if isinstance(instr, StoreRegToGlb):
        break
  
    for index in range(num_remove_instrs - 1):
      self._instrs.pop(-1)
