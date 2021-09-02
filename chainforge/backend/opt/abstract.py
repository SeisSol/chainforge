from chainforge.common import Context
from chainforge.backend.instructions import AbstractInstruction
from abc import ABC, abstractmethod
from typing import List


class AbstractOptStage(ABC):
  def __init__(self,
               context: Context):
    self._context = context
    
  @abstractmethod
  def apply(self) -> None:
    pass


class AbstractTransformer(AbstractOptStage):
  def __init__(self,
               context: Context,
               instructions: List[AbstractInstruction]):
    super(AbstractTransformer, self).__init__(context)
    
    self._context = context
    self._instrs = instructions
    
  @abstractmethod
  def apply(self) -> None:
    pass

  def get_instructions(self) -> List[AbstractInstruction]:
    return self._instrs
  
  def _print_instr(self) -> None:
    for index, instr in enumerate(self._instrs):
      print(f'{index}:   {instr}')
