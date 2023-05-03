from typing import List
from chainforge.backend.instructions import StoreRegToGlb
from chainforge.backend.instructions import ClearRegisters
from chainforge.backend.instructions import StoreGlbToReg
from .abstract import AbstractTransformer, Context, AbstractInstruction


class DataDependencyOpt(AbstractTransformer):
  '''
  This optimization removes a write-after-read dependency which commonly
  occurs at the end of a fused kernel i.e., when `StoreRegToGlb`
  instruction is getting produced by `C = * A x B + beta * C` operation,
  when `beta != 0`.

  The optimization 1) traverses instructions from bottom to top,
  2) finds the last `StoreRegToGlb` and `ClearRegisters` operations,
  3) sets `beta` parameter of `StoreRegToGlb` to zero, 4) replaces
  `ClearRegisters` with `StoreGlbToReg` operation with the `beta`
  parameter given in the original `StoreRegToGlb` operation.
  '''

  def __init__(self,
               context: Context,
               instructions: List[AbstractInstruction],
               num_threads: int):
    super(DataDependencyOpt, self).__init__(context, instructions)
    self._reg_to_glb_index = None
    self._clear_reg_index = None
    self._num_threads = num_threads

  def _find_candidates(self):
    for index, inst in reversed(list(enumerate(self._instrs))):
      if isinstance(inst, StoreRegToGlb):
        self._reg_to_glb_index = index

      if isinstance(inst, ClearRegisters):
        self._clear_reg_index = index
        break

    found = False
    if self._reg_to_glb_index and self._clear_reg_index:
      if self._reg_to_glb_index > self._clear_reg_index:
        store_op = self._instrs[self._reg_to_glb_index]
        beta = store_op.get_beta()
        if isinstance(beta, float) and beta != 0.0:
          found = True

    return found

  def apply(self) -> None:
    found = self._find_candidates()
    if found:
      store_instr = self._instrs[self._reg_to_glb_index]
      beta = store_instr.get_beta()
      store_instr.set_beta(0.0)

      clear_reg_instr = self._instrs.pop(self._clear_reg_index)
      clear_reg_instr.unregister()

      substitute_instr = StoreGlbToReg(context=self._context,
                                       src=store_instr.get_dest(),
                                       dest=clear_reg_instr.get_src(),
                                       beta=beta,
                                       num_threads=self._num_threads)

      self._instrs.insert(self._clear_reg_index, substitute_instr)
