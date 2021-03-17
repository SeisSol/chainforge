from .abstract_builder import AbstractBuilder
from chainforge.common.vm import VM
from chainforge.backend.scopes import Scopes, Symbol
from chainforge.common.matrix import Matrix
from chainforge.backend.symbol import SymbolType
from chainforge.backend.instructions import GetElementPtr
from chainforge.backend.exceptions import InternalError


class GetElementPtrBuilder(AbstractBuilder):
  def __init__(self, vm: VM, scopes: Scopes):
    super(GetElementPtrBuilder, self).__init__(vm, scopes)

  def build(self, src: Symbol):
    self._reset()
    if src.stype != SymbolType.Batch:
      raise InternalError("src operand is not in a batch")

    if issubclass(Matrix, type(src.obj)):
      raise InternalError(f'src operand is not a matrix. Instead: {type(src.obj)}')

    dest = Symbol(name=f'glb{src.name}',
                  stype=SymbolType.Global,
                  obj=src.obj)

    self._scopes.add_symbol(dest)
    self._instructions.append(GetElementPtr(self._vm, src, dest))
    src.add_user(self)
