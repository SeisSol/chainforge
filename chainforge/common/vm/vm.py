from .arch_lexic import AbstractArchLexic, lexic_factory
from .hw_descr import HwDecription, hw_descr_factory
from chainforge.common.basic_types import FloatingPointType
from math import ceil
from typing import Type


class VM:
  def __init__(self,
               hw_descr: HwDecription,
               basic_arch_lexic: Type[AbstractArchLexic],
               fp_type: FloatingPointType):
    self.hw_descr = hw_descr
    self.lexic = basic_arch_lexic
    self.fp_type = fp_type


  def set_fp_type(self, fp_type: FloatingPointType):
    self.fp_type = fp_type

  def fp_as_str(self):
    return FloatingPointType.as_str(self.fp_type)

  def align(self, num):
    return ceil(num / self.hw_descr.vec_unit_length) * self.hw_descr.vec_unit_length


def vm_factory(name: str,
               sub_name: str,
               fp_type: FloatingPointType):

  descr = hw_descr_factory(name, sub_name)
  lexic = lexic_factory(name)
  return VM(hw_descr=descr,
            basic_arch_lexic=lexic,
            fp_type=fp_type)
