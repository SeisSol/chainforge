from .shr_mem_loader import ExtendedPatchLoader, ExactPatchLoader
from .shr_trans_mem_loader import ExtendedTransposePatchLoader, ExactTransposePatchLoader
from .abstract_loader import ShrMemLoaderType, AbstractShrMemLoader
from chainforge.common.matrix import Matrix
from chainforge.backend.exceptions import InternalError
from math import ceil


def shm_mem_loader_factory(vm, dest, src, shr_mem, num_threads, load_and_transpose=False):
  params = {'vm': vm,
            'dest': dest,
            'src': src,
            'shr_mem': shr_mem,
            'num_threads': num_threads,
            'load_and_transpose': load_and_transpose}


  if not isinstance(src.obj, Matrix):
    raise InternalError('shm-factory: `src` operand is not a matrix')

  # Use an extended loader if the tail of a active threads can touch the next column
  # Otherwise, use an exact one
  num_loads_per_column = ceil(src.obj.get_actual_num_rows() / num_threads) * num_threads

  if src.obj.num_rows > num_loads_per_column:
    if load_and_transpose:
      return ExactTransposePatchLoader(**params)
    else:
      return ExactPatchLoader(**params)
  else:
    if load_and_transpose:
      return ExtendedTransposePatchLoader(**params)
    else:
      return ExtendedPatchLoader(**params)
