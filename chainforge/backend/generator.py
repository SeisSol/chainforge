from typing import List, Union, Type
from copy import deepcopy
import hashlib
from chainforge.common import GemmDescr
from chainforge.common import Context
from chainforge.common import Addressing, GeneralLexicon
from chainforge.common.aux import get_extra_offset_name
from .data_types import ShrMemObject, RegMemObject
from .opt import OptimizationStage
from .scopes import Scopes
from .symbol import Symbol, SymbolType
from .instructions import AbstractInstruction
from .instructions.builders.kernels import kernel_factory
from .instructions.builders.kernels import KernelType
from .writer import Writer
from .thread_block_policies import AbstractThreadBlockPolicy, SimpleThreadBlockPolicy
from .exceptions import GenerationError


class Generator:
  NAME_ENCODING_LENGTH = 10

  def __init__(self,
               gemm_list: List[GemmDescr],
               context: Context,
               kernel_type: KernelType = KernelType.AUTO,
               thread_block_policy_type: Type[AbstractThreadBlockPolicy] = SimpleThreadBlockPolicy):
    self.gemm_list: List[GemmDescr] = gemm_list
    self._context: Context = context
    self._kernel_type: KernelType = kernel_type
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
    self._accumulator_size: int = 0

    self._shr_mem_obj: Union[ShrMemObject, None] = None
    self._register_array_obj: Union[RegMemObject, None] = None

    self._ir: List[AbstractInstruction] = []

    self._check_consistency_with_user_options()
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

    self._emit_ir()
    opt = OptimizationStage(context=self._context,
                            shr_mem=self._shr_mem_obj,
                            instructions=self._ir,
                            num_threads=self._num_threads)
    opt.optimize()
    self._ir = opt.get_instructions()
    self._deduce_mults_per_block()

    self._generate_kernel()
    self._generate_launcher()
    self._generate_header()

  def _generate_kernel(self):
    writer = Writer()
    proto = self._generate_kernel_proto()
    with writer.block(f'{proto}'):
      self._write_kernel_meta_data(writer)

      writer(f'unsigned {GeneralLexicon.BATCH_ID_NAME} = {self._get_2d_block_id()};')
      with writer.block(f'if ({self._get_element_size_guard()})'):
        with writer.block(f'if ({self._get_flag_guard(writer)})'):
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
    lexic = self._context.get_vm().lexic
    with writer.block(f'{proto}'):
      writer(f'{lexic.dim3_type} block({self._num_threads}, {mults_per_block}, 1);')
      num_blocks = f'({GeneralLexicon.NUM_ELEMENTS} + {mults_per_block} - 1) / {mults_per_block}'
      writer(f'{lexic.dim3_type} grid({num_blocks}, 1, 1);')

      if_stream_exists = f'({GeneralLexicon.STREAM_PTR_STR} != nullptr)'
      stream_obj = f'static_cast<{lexic.stream_type}>({GeneralLexicon.STREAM_PTR_STR})'
      writer(f'{lexic.stream_type} stream = {if_stream_exists} ? {stream_obj} : 0;')

      args = self._generate_kernel_base_args()
      args = ', '.join(args)
      kernel_name = f'kernel_{self._base_kernel_name}'
      call_site = lexic.get_launch_code(func_name=kernel_name,
                                        grid='grid',
                                        block='block',
                                        stream='stream',
                                        func_params=args)
      writer(f'{call_site};')
      writer('CHECK_ERR;')
    self._launcher = writer.get_src()

  def _generate_header(self):
    self._header = f'{self._generate_launcher_proto(with_defaults=True)};\n'

  def _emit_ir(self):
    builder = kernel_factory(self._context,
                             self._scopes,
                             self.gemm_list,
                             self._kernel_type)
    builder.build()
    self._ir = builder.get_instructions()
    self._num_threads = builder.get_num_threads()
    self._accumulator_size = builder.get_accumulator_size()
    self._register_array_obj = builder.get_reg_array_obj()
    self._shr_mem_obj = builder.get_shr_mem_obj()
    self._kernel_type = builder.get_selected_kernel_type()

  def _deduce_mults_per_block(self):
    policy = self._thread_block_policy_type(self._context,
                                            self._kernel_type,
                                            self._shr_mem_obj.get_size_per_mult(),
                                            self._num_threads)
    num_mults_per_block = policy.get_num_mults_per_block()
    self._shr_mem_obj.set_mults_per_block(num_mults_per_block)

  def get_kernel(self):
    return self._kernel

  def get_launcher(self):
    return self._launcher

  def get_header(self):
    return self._header

  def _check_consistency_with_user_options(self):
    user_options = self._context.get_user_options()
    for gemm in self.gemm_list:
      if not gemm.is_strict_math() == user_options.exact_contraction_length:
        msg = 'gemm list is not consistent with user options. '
        msg += f'`strict_math` in gemm descr. set to {gemm.is_strict_math()}, '
        msg += f'but `exact_contraction_length` is set to {user_options.exact_contraction_length}'
        raise RuntimeError(msg)

  def _name_operands(self, gemm_list: List[GemmDescr]):
    tmp_counter = 0
    op_counter = 'A'
    tmp_base_name = 'tmp'

    self._matrix_list = []
    for gemm in gemm_list:
      local_list = [gemm.mat_a, gemm.mat_b, gemm.mat_c]

      # NOTE: to be on the safe side we init all matrix names with None
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
    long_name = []
    for item in global_symbols:
      long_name.append(item.obj.gen_descr())

    for gemm in self.gemm_list:
      long_name.extend([
        str(gemm.alpha),
        str(gemm.beta),
        str(gemm.trans_a),
        str(gemm.trans_b)
      ])

    result = hashlib.md5(', '.join(long_name).encode())
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
    for matrix in glb_matrices:
      writer(f'// {matrix.obj.gen_descr()}')

    writer.new_line()
    for item in self.gemm_list:
      writer(f'// {item}')
    writer.new_line()

  def _generate_scalar_param_list(self, with_types=True):
    scalar_type = self._context.fp_as_str() if with_types else ''
    last_gemm = self.gemm_list[-1]
    params = []
    if not isinstance(last_gemm.alpha, (float, int)):
      name = self._get_scalar_name(last_gemm.alpha, GeneralLexicon.ALPHA_SYMBOL_NAME)
      params.append(f'{scalar_type} {name}')

    if not isinstance(last_gemm.beta, (float, int)):
      name = self._get_scalar_name(last_gemm.beta, GeneralLexicon.BETA_SYMBOL_NAME)
      params.append(f'{scalar_type} {name}')

    return params

  def _generate_base_params_list(self, symbol_list, with_types=True, with_defaults=False):
    fp_as_str = self._context.fp_as_str()
    params = []
    for symbol in symbol_list:
      ptr_type = Addressing.addr2ptr_type(symbol.obj.addressing)
      batch_type = f'{fp_as_str}{ptr_type}' if with_types else ''
      offset_type = 'unsigned' if with_types else ''
      params.extend([f'{batch_type} {symbol.name}',
                     f'{offset_type} {get_extra_offset_name(symbol)}'])

    batch_size_type = 'size_t' if with_types else ''
    params.append(f'{batch_size_type} {GeneralLexicon.NUM_ELEMENTS}')

    flags_type = 'unsigned*' if with_types else ''
    default_flags_value = '= nullptr' if with_defaults else ''
    params.append(f'{flags_type} {GeneralLexicon.FLAGS_NAME} {default_flags_value}')
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

    lexic = self._context.get_vm().lexic

    launch_bounds = lexic.get_launch_bounds(total_num_threads_per_block)
    return f'{lexic.kernel_type} {launch_bounds} kernel_{self._base_kernel_name}({params})'

  def _generate_launcher_proto(self, with_defaults=True):
    global_symbols = self._scopes.get_global_scope().values()
    params = self._generate_scalar_param_list()

    params.extend(self._generate_base_params_list(symbol_list=global_symbols,
                                                  with_types=True,
                                                  with_defaults=with_defaults))

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

    args.append(f'{GeneralLexicon.FLAGS_NAME}')
    args.append(f'{GeneralLexicon.STREAM_PTR_STR}')
    args = ', '.join(args)
    return f'launcher_{self._base_kernel_name}({args});'

  def generate_call_site(self,
                         mat_name_map,
                         offset_name_map,
                         alpha,
                         beta,
                         num_element,
                         flags=None,
                         stream=None):
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

    # add flags
    if flags:
      args.append(flags)

    # add streams
    if stream:
      args.append(stream)

    args = ', '.join(args)
    return f'launcher_{self._base_kernel_name}({args});'

  def _get_2d_block_id(self):
    lexic = self._context.get_vm().lexic
    return f'{lexic.thread_idx_y} + {lexic.block_dim_y} * {lexic.block_idx_x}'

  def _get_element_size_guard(self):
    return f'{GeneralLexicon.BATCH_ID_NAME} < {GeneralLexicon.NUM_ELEMENTS}'

  def _get_flag_guard(self, writer):
    writer(f'bool isFlagsProvided = ({GeneralLexicon.FLAGS_NAME} != nullptr);')
    flag_value = f'static_cast<bool>({GeneralLexicon.FLAGS_NAME}[{GeneralLexicon.BATCH_ID_NAME}])'
    writer(f'bool allowed = isFlagsProvided ? {flag_value} : true;')
    return 'allowed'
