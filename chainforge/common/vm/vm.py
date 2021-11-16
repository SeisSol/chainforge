from typing import Type
from .arch_lexic import AbstractArchLexic, lexic_factory
from .hw_descr import HwDecription, hw_descr_factory


class VM:
  def __init__(self,
               hw_descr: HwDecription,
               basic_arch_lexic: Type[AbstractArchLexic]):
    self.hw_descr = hw_descr
    self.lexic = basic_arch_lexic


def vm_factory(arch: str, backend: str):
  descr = hw_descr_factory(arch, backend)
  lexic = lexic_factory(backend)
  return VM(hw_descr=descr, basic_arch_lexic=lexic)
