from abc import ABC
from ..abstract_instruction import AbstractInstruction
from typing import List
from chainforge.common import Context


class AbstractBuilder(ABC):
  def __init__(self, context, scopes):
    if not isinstance(context, Context):
      raise RuntimeError(f'received wrong type, expected Context, given {type(context)}')
    
    self._context = context
    #self._vm = context.get_vm()
    self._scopes = scopes
    self._instructions = []

  def get_instructions(self) -> List[AbstractInstruction]:
    return self._instructions

  def _reset(self):
    self._instructions = []
