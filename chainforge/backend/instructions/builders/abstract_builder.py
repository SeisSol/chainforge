from abc import ABC
from ..abstract_instruction import AbstractInstruction
from typing import List


class AbstractBuilder(ABC):
  def __init__(self, vm, scopes):
    self._vm = vm
    self._scopes = scopes
    self._instructions = []

  def get_instructions(self) -> List[AbstractInstruction]:
    return self._instructions

  def _reset(self):
    self._instructions = []
