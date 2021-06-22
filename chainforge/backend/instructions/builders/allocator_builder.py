from .abstract_builder import AbstractBuilder
from chainforge.common.vm import VM
from chainforge.backend.scopes import Scopes
from chainforge.backend.symbol import Symbol, SymbolType
from chainforge.backend.instructions.allocate import RegisterAlloc, ShrMemAlloc
from chainforge.backend.data_types import ShrMemObject, RegMemObject
from abc import abstractmethod
from typing import Union


class AbstractAllocBuilder(AbstractBuilder):
  def __init__(self, vm: VM, scopes: Scopes):
    super(AbstractAllocBuilder, self).__init__(vm, scopes)
    self._obj = None

  @abstractmethod
  def _name_new_symbol(self):
    pass

  def get_resultant_obj(self):
    if not self._obj:
      raise NotImplementedError
    return self._obj


class ShrMemAllocBuilder(AbstractAllocBuilder):
  def __init__(self,
               vm: VM,
               scopes: Scopes):
    super(ShrMemAllocBuilder, self).__init__(vm, scopes)
    self._counter = 0

  def build(self, size: Union[int, None] = None):
    self._reset()
    name = self._name_new_symbol()
    self._obj = ShrMemObject(name, size)
    dest = Symbol(name=name,
                  stype=SymbolType.SharedMem,
                  obj=self._obj)

    self._scopes.add_symbol(dest)
    self._instructions.append(ShrMemAlloc(self._vm, dest, size))

  def _name_new_symbol(self):
    name = f'shrmem{self._counter}'
    self._counter += 1
    return name


class RegistersAllocBuilder(AbstractAllocBuilder):

  def __init__(self,
               vm: VM,
               scopes: Scopes):
    super(RegistersAllocBuilder, self).__init__(vm, scopes)
    self._counter = 0

  def build(self, size: int, init_value: Union[float, None] = None):
    self._reset()
    name = self._name_new_symbol()
    self._obj = RegMemObject(name, size)
    dest = Symbol(name,
                  SymbolType.Register,
                  self._obj)

    self._scopes.add_symbol(dest)
    self._instructions.append(RegisterAlloc(self._vm, dest, size, init_value))

  def _name_new_symbol(self):
    name = f'reg{self._counter}'
    self._counter += 1
    return name
