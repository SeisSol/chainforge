from .data_types import ShrMemObject, RegMemObject
from chainforge.common import GemmDescr
from chainforge.common import VM
from chainforge.common import Addressing, GeneralLexicon
from chainforge.common.aux import get_extra_offset_name
from .opt import Optimizer
from .scopes import Scopes
from .symbol import Symbol, SymbolType
from .instructions import AbstractInstruction
from .instructions import GetElementPtrBuilder, GemmBuilder
from .instructions import ShrMemAllocBuilder, RegistersAllocBuilder
from .writer import Writer
from .thread_block_policies import AbstractThreadBlockPolicy, SimpleThreadBlockPolicy
from .exceptions import GenerationError
from typing import List, Union
from copy import deepcopy
import hashlib
from typing import Type


class Generator:
  NAME_ENCODING_LENGTH = 8

  def __init__(self,
               gemm_list: List[GemmDescr],
               vm: VM,
               thread_block_policy_type: Type[AbstractThreadBlockPolicy] = SimpleThreadBlockPolicy):
    self.gemm_list: List[GemmDescr] = gemm_list
    self._vm: VM = vm
    self._thread_block_policy_type: Type[AbstractThreadBlockPolicy] = thread_block_policy_type
    self._base_kernel_name: Union[str, None] = None

    self._kernel = None
    self._launcher = None
    self._header = None

    self._matrix_list = None
    self._tmp_list = None
    self._scopes: Scopes = Scopes()
    self._is_registerd: bool = False

    self._num_threads: int = 0
    self._num_active_threads: int = 0
    self._accumulator_size: int = 0

    self._shr_mem_obj: Union[ShrMemObject, None] = None
    self._register_array_obj: Union[RegMemObject, None] = None

    self._ir: List[AbstractInstruction] = []

    self._name_operands(self.gemm_list)

  def set_kernel_name(self, name):
    self._base_kernel_name = name

  def register(self):
    self._collect_tmp_matrices()
    self._populate_global_scope()
    if not self._base_kernel_name:
      self._generate_kernel_name()
    self._is_registerd = True

  def generate(self):
    if not self._is_registerd:
      self.register()

    self._deduce_num_threads()
    self._deduce_accumulator_size()
    self._emit_ir()
    opt = Optimizer(shr_mem=self._shr_mem_obj, instructions=self._ir)
    opt.optimize()
    self._deduce_mults_per_block()

    self._generate_kernel()
    self._generate_launcher()
    self._generate_header()

  def _generate_kernel(self):
    writer = Writer()
    proto = self._generate_kernel_proto()
    with writer.block(f'{proto}'):
      self._write_kernel_meta_data(writer)

      with writer.block(f'if (({self._get_2d_block_id()}) < {GeneralLexicon.NUM_ELEMENTS})'):
        for instruction in self._ir:
          if instruction.is_ready():
            instruction.gen_code(writer)
          else:
            raise GenerationError(f'instr is not ready to be generated: {instruction}')

    self._kernel = writer.get_src()

  def _generate_launcher(self):
    writer = Writer()
    proto = self._generate_launcher_proto(with_defaults=False)
    mults_per_block = self._shr_mem_obj.get_mults_per_block()
    with writer.block(f'{proto}'):
      writer(f'{self._vm.lexic.dim3_type} block({self._num_threads}, {mults_per_block}, 1);')
      num_blocks = f'({GeneralLexicon.NUM_ELEMENTS} + {mults_per_block} - 1) / {mults_per_block}'
      writer(f'{self._vm.lexic.dim3_type} grid({num_blocks}, 1, 1);')

      if_stream_exists = f'({GeneralLexicon.STREAM_PTR_STR} != nullptr)'
      stream_obj = f'static_cast<cudaStream_t>({GeneralLexicon.STREAM_PTR_STR})'
      writer(f'{self._vm.lexic.stream_name} stream = {if_stream_exists} ? {stream_obj} : 0;')

      args = self._generate_kernel_base_args()
      args = ', '.join(args)
      kernel_name = f'kernel_{self._base_kernel_name}'
      call_site = self._vm.lexic.get_launch_code(func_name=kernel_name,
                                                 grid='grid',
                                                 block='block',
                                                 stream='stream',
                                                 func_params=args)
      writer(f'{call_site};')
      writer('CHECK_ERR;')
    self._launcher = writer.get_src()

  def _generate_header(self):
    self._header = f'{self._generate_launcher_proto(with_defaults=True)};\n'

  def _deduce_num_threads(self):
    for gemm in self.gemm_list:
      num_threads, num_active_threads = gemm.get_num_threads(self._vm)
      self._num_threads = max(num_threads, self._num_threads)
      self._num_active_threads = max(num_active_threads, self._num_active_threads)

  def _deduce_accumulator_size(self):
    for gemm in self.gemm_list:
      local_acc_size = gemm.get_accumulator_size()
      self._accumulator_size = max(self._accumulator_size, local_acc_size)

  def _emit_ir(self):
    # find local data from batches
    builder = GetElementPtrBuilder(self._vm, self._scopes)
    self._scopes.add_scope()
    for symbol in self._scopes.get_global_scope().values():
      builder.build(symbol)
      self._ir.extend(builder.get_instructions())

    # allocate registers
    builder = RegistersAllocBuilder(self._vm, self._scopes)
    builder.build(self._accumulator_size, 0.0)
    self._register_array_obj = builder.get_resultant_obj()
    self._ir.extend(builder.get_instructions())

    # allocate shared memory
    builder = ShrMemAllocBuilder(self._vm, self._scopes)
    builder.build(size=None)
    self._shr_mem_obj = builder.get_resultant_obj()
    self._ir.extend(builder.get_instructions())

    self._scopes.add_scope()
    # generate GEMM and store operations
    builder = GemmBuilder(self._vm,
                          self._scopes,
                          self._scopes.get_symbol(self._register_array_obj),
                          self._scopes.get_symbol(self._shr_mem_obj),
                          self._num_threads)

    for gemm_descr in self.gemm_list:
      builder.build(op1=self._scopes.get_symbol(gemm_descr.mat_a),
                    op2=self._scopes.get_symbol(gemm_descr.mat_b),
                    dest_obj=gemm_descr.mat_c,
                    descr=gemm_descr)
      self._ir.extend(builder.get_instructions())

  def _deduce_mults_per_block(self):
    policy = self._thread_block_policy_type(self._vm,
                                            self._shr_mem_obj.get_size_per_mult(),
                                            self._num_threads)
    num_mults_per_block = policy.get_num_mults_per_block()
    self._shr_mem_obj.set_mults_per_block(num_mults_per_block)

  def get_kernel(self):
    return self._kernel

  def get_launcher(self):
    return  self._launcher

  def get_header(self):
    return self._header

  def _name_operands(self, gemm_list: List[GemmDescr]):
    tmp_counter = 0
    op_counter = 'A'
    tmp_base_name = 'tmp'

    self._matrix_list = []
    for gemm in gemm_list:
      local_list = [gemm.mat_a, gemm.mat_b, gemm.mat_c]

      # NOTE: to be on the sage side we init all matrix names with None
      for matrix in local_list:
        matrix.name = None

      # gather all matrices
      self._matrix_list.extend(local_list)

    for matrix in self._matrix_list:
      # if matrix name is not None
      if not matrix.name:
        if matrix.is_tmp:
          matrix.name = f'{tmp_base_name}{tmp_counter}'
          tmp_counter += 1
        else:
          matrix.name = f'{op_counter}'
          op_counter = chr(ord(op_counter) + 1)

  def _collect_tmp_matrices(self):
    self._tmp_list = []
    for matrix in self._matrix_list:
      if matrix.is_tmp and matrix not in self._tmp_list:
        self._tmp_list.append(matrix)

  def _populate_global_scope(self):
    """
    Add non-tmp matrices to the global scope
    :return:
    """
    for matrix in self._matrix_list:
      if matrix not in self._tmp_list:
        self._scopes.add_to_global(Symbol(obj=matrix,
                                          name=matrix.name,
                                          stype=SymbolType.Batch))

  def _generate_kernel_name(self):
    global_symbols = self._scopes.get_global_scope().values()
    glb2str = []
    for item in global_symbols:
      glb2str.append(item.obj.gen_descr())

    result = hashlib.md5(', '.join(glb2str).encode())
    md5encoding = result.hexdigest()
    self._base_kernel_name = f'cf_gemms_{md5encoding[:Generator.NAME_ENCODING_LENGTH]}'

  def get_base_name(self):
    return self._base_kernel_name

  def _get_scalar_name(self, scalar, default_name):
    scalar_type = type(scalar)
    is_pritable = scalar_type.__str__ is not object.__str__
    is_string = scalar_type == str
    return scalar if is_pritable or is_string else default_name

  def _write_kernel_meta_data(self, writer):
    writer('// meta data:')
    glb_matrices = self._scopes.get_global_scope().values()
    delimiter = '; '
    for matrix in glb_matrices:
      writer(f'// {matrix.obj.gen_descr(delimiter)}')

    writer.new_line()
    for item in self.gemm_list:
      writer(f'// {item}')
    writer.new_line()

  def _generate_scalar_param_list(self, with_types=True):
    scalar_type = self._vm.fp_as_str() if with_types else ''
    last_gemm = self.gemm_list[-1]
    params = []
    if not isinstance(last_gemm.alpha, (float, int)):
      name = self._get_scalar_name(last_gemm.alpha, GeneralLexicon.ALPHA_SYMBOL_NAME)
      params.append(f'{scalar_type} {name}')

    if not isinstance(last_gemm.beta, (float, int)):
      name = self._get_scalar_name(last_gemm.beta, GeneralLexicon.BETA_SYMBOL_NAME)
      params.append(f'{scalar_type} {name}')

    return params

  def _generate_base_params_list(self, symbol_list, with_types=True):
    params = []
    for symbol in symbol_list:
      ptr_type = Addressing.addr2ptr_type(symbol.obj.addressing)
      batch_type = f'{self._vm.fp_as_str()}{ptr_type}' if with_types else ''
      offset_type = 'unsigned' if with_types else ''
      params.extend([f'{batch_type} {symbol.name}',
                     f'{offset_type} {get_extra_offset_name(symbol)}'])

    batch_size_type = 'size_t' if with_types else ''
    params.append(f'{batch_size_type} {GeneralLexicon.NUM_ELEMENTS}')
    return params

  def _generate_kernel_base_args(self):
    global_symbols = self._scopes.get_global_scope().values()
    args = self._generate_scalar_param_list(with_types=False)
    args.extend(self._generate_base_params_list(global_symbols, with_types=False))
    return args

  def _generate_kernel_proto(self):
    global_symbols = self._scopes.get_global_scope().values()
    params = self._generate_scalar_param_list()

    params.extend(self._generate_base_params_list(symbol_list=global_symbols,
                                                  with_types=True))
    params = ', '.join(params)
    total_num_threads_per_block = self._num_threads * self._shr_mem_obj.get_mults_per_block()
    launch_bounds = self._vm.lexic.get_launch_bounds(total_num_threads_per_block)
    return f'{self._vm.lexic.kernel_type} kernel_{self._base_kernel_name}({params})'

  def _generate_launcher_proto(self, with_defaults=True):
    global_symbols = self._scopes.get_global_scope().values()
    params = self._generate_scalar_param_list()

    params.extend(self._generate_base_params_list(symbol_list=global_symbols,
                                                  with_types=True))

    default_value = ' = nullptr' if with_defaults else ''
    params.append(f'void* {GeneralLexicon.STREAM_PTR_STR}{default_value}')
    params = ', '.join(params)
    return f'void launcher_{self._base_kernel_name}({params})'

  def default_generate_call_site(self):
    if not self._is_registerd:
      raise RuntimeError('generator is not registered. Call register first.')
    symbols = deepcopy(list(self._scopes.get_global_scope().values()))
    for item in symbols:
      if item.obj.alias:
        item.name = item.obj.alias

    args = self._generate_scalar_param_list(with_types=False)
    args.extend(self._generate_base_params_list(symbol_list=symbols,
                                                with_types=False))

    args.append(f'{GeneralLexicon.STREAM_PTR_STR}')
    args = ', '.join(args)
    return f'launcher_{self._base_kernel_name}({args});'

  def generate_call_site(self, mat_name_map, offset_name_map, alpha, beta, num_element, stream=None):
    args = []

    # add scalars
    scalars = [alpha, beta]
    for scalar in scalars:
      if not isinstance(scalar, float):
        args.append(scalar)

    # add matrices
    symbols = list(self._scopes.get_global_scope().values())
    for symbol in symbols:
      if symbol.obj.alias in mat_name_map:
        args.append(mat_name_map[symbol.obj.alias])
        args.append(offset_name_map[symbol.obj.alias])

    # add num. elements
    args.append(num_element)

    # add streams
    if stream:
      args.append(stream)

    args = ', '.join(args)
    return f'launcher_{self._base_kernel_name}({args});'

  def _get_2d_block_id(self):
    return f'{self._vm.lexic.threadIdx_y} + {self._vm.lexic.blockDim_y} * {self._vm.lexic.blockIdx_x}'
