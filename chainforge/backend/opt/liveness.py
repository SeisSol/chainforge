from chainforge.backend.symbol import Symbol
from chainforge.backend.instructions import AbstractInstruction
from chainforge.backend.instructions import Gemm, StoreRegToShr
from chainforge.backend.instructions.loaders import AbstractShrMemLoader
from chainforge.backend.symbol import SymbolType
from typing import List, Dict, Set
from copy import copy
from collections import OrderedDict


class LivenessAnalysis:
  def __init__(self, instructions: List[AbstractInstruction]):
    self._instrs: List[AbstractInstruction] = instructions
    self._map: Dict[int, Set[Symbol]] = OrderedDict()

  def apply(self) -> Dict[int, Set[Symbol]]:
    self._map = {len(self._instrs): set()}

    for index, instr in reversed(list(enumerate(self._instrs))):
      self._map[index] = copy(self._map[index + 1])
      if isinstance(instr, Gemm):
        self._check_use(index, instr)
      elif isinstance(instr, (StoreRegToShr, AbstractShrMemLoader)):
        self._check_define(index, instr)

    return OrderedDict(reversed(list(self._map.items())))

  def _check_use(self, index, instr) -> None:
    operands = [instr.get_op1(), instr.get_op2()]
    for operand in operands:
      if operand.stype == SymbolType.SharedMem:
        self._map[index].add(operand)

  def _check_define(self, index, instr) -> None:
    self._map[index].remove(instr.get_dest())
