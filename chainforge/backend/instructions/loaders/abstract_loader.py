from abc import abstractmethod
from typing import Union
import enum
from chainforge.common.matrix import Matrix
from chainforge.backend.instructions import AbstractShrMemWrite
from chainforge.backend.symbol import SymbolType, Symbol
from chainforge.backend.exceptions import InternalError


class ShrMemLoaderType(enum.Enum):
  NOT_TRANSPOSED = 0
  TRANSPOSED = 1


class AbstractShrMemLoader(AbstractShrMemWrite):
  def __init__(self, **kwargs):
    super(AbstractShrMemLoader, self).__init__(kwargs['context'])
    self._dest = kwargs['dest']
    self._src = kwargs['src']
    self._shr_mem = kwargs['shr_mem']
    self._num_threads = kwargs['num_threads']
    self._load_and_transpose = kwargs['load_and_transpose']
    self._manual_unroll_threshold = 4

    self._check()
    self._lid_dim: Union[int, None] = None
    self._align_shm_volume: Union[int, None] = None
    self._matrix: Matrix = self._src.obj

    self._dest.add_user(self)
    self._src.add_user(self)
    self._shr_mem.add_user(self)
    self._is_ready: bool = False

  def gen_code(self, writer) -> None:
    writer.new_line()
    lhs = f'{self._fp_as_str}* {self._vm.lexic.restrict_kw} {self._dest.name}'
    rhs = f'{self._shr_mem.name}[{self._shr_mem_offset}]'
    writer(f'{lhs} = &{rhs};')

  def get_src(self) -> Symbol:
    return self._src

  def get_dest(self) -> Symbol:
    return self._dest

  @abstractmethod
  def get_loader_type(self) -> ShrMemLoaderType:
    pass

  def _check(self) -> None:
    if self._src.stype != SymbolType.Global:
      raise InternalError('shr-load: `src` operand is not in global mem.')

    if not isinstance(self._src.obj, Matrix):
      raise InternalError(f'shr-load: `src` operand is not a matrix, instead: {self._src.obj}')

    if self._dest.stype != SymbolType.SharedMem:
      raise InternalError('shr-load: `dest` operand is not in shr. mem.')

    if not isinstance(self._dest.obj, Matrix):
      raise InternalError(f'shr-load: `dest` operand is not a matrix, instead: {self._dest.obj}')
