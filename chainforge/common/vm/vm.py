from .arch_lexic import AbstractArchLexic, lexic_factory
from .hw_descr import HwDecription, hw_descr_factory
from typing import Type


class VM:
  def __init__(self,
               hw_descr: HwDecription,
               basic_arch_lexic: Type[AbstractArchLexic]):
    self.hw_descr = hw_descr
    self.lexic = basic_arch_lexic


def vm_factory(name: str, sub_name: str):
  descr = hw_descr_factory(name, sub_name)
  lexic = lexic_factory(name)
  return VM(hw_descr=descr, basic_arch_lexic=lexic)
