from abc import ABC
from typing import List
from chainforge.common import Context
from ..abstract_instruction import AbstractInstruction


class AbstractBuilder(ABC):
  def __init__(self, context, scopes):
    if not isinstance(context, Context):
      raise RuntimeError(f'received wrong type, expected Context, given {type(context)}')

    self._context = context
    self._scopes = scopes
    self._instructions = []

  def get_instructions(self) -> List[AbstractInstruction]:
    return self._instructions

  def _reset(self):
    self._instructions = []
