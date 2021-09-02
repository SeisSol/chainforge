from chainforge.backend.exceptions import GenerationError
from .context import Context
from .basic_types import DataFlowDirection, FloatingPointType


class GemmDescr:
  def __init__(self,
               trans_a,
               trans_b,
               a,
               b,
               c,
               alpha=1.0,
               beta=0.0,
               strict_match: bool = False):
    self.trans_a = trans_a
    self.trans_b = trans_b
    self.mat_a = a
    self.mat_a.set_data_flow_direction(DataFlowDirection.SOURCE)

    self.mat_b = b
    self.mat_b.set_data_flow_direction(DataFlowDirection.SOURCE)

    self.mat_c = c
    self.mat_c.set_data_flow_direction(DataFlowDirection.SINK)

    self.alpha = alpha
    self.beta = beta

    self._m = None
    self._n = None
    self._k = None
    self._strict_match = strict_match

    self._check()
    self._analyze()

  def _analyze(self):
    if self.trans_a:
      self._m = self.mat_a.get_actual_num_cols()
      self._k = self.mat_a.get_actual_num_rows()
    else:
      self._m = self.mat_a.get_actual_num_rows()
      self._k = self.mat_a.get_actual_num_cols()

    if self.trans_b:
      self._n = self.mat_b.get_actual_num_rows()
    else:
      self._n = self.mat_b.get_actual_num_cols()

  def get_num_threads(self, context: Context):
    num_threads = context.align(num=self._m)
    return num_threads, self._m

  def get_accumulator_size(self):
    return self._n

  def is_strict_math(self):
    return self._strict_match

  def __str__(self):
    suffix_a = '^T' if self.trans_a else ''
    suffix_b = '^T' if self.trans_b else ''
    op1 = f'{self.alpha} * {self.mat_a}{suffix_a} x {self.mat_b}{suffix_b}'
    op2 = '' if self.beta == 0 else f' + {self.beta} * {self.mat_c}'
    return f'{self.mat_c} = {op1}{op2}'

  def _check(self):
    try:
      # check whether C and A match each other
      if self.trans_a:
        if self.mat_c.get_actual_num_rows() != self.mat_a.get_actual_num_cols():
          raise GenerationError("Cannot generate a matrix multiplication "
                                "with given parameters. Matrix C and A (Trans) do not match")
      else:
        if self.mat_c.get_actual_num_rows() != self.mat_a.get_actual_num_rows():
          raise GenerationError("Cannot generate a matrix multiplication "
                                "with given parameters. Matrix C and A (NoTrans) do not match")

      # check whether C and B match each other
      if self.trans_b:
        if self.mat_c.get_actual_num_cols() != self.mat_b.get_actual_num_rows():
          raise GenerationError("Cannot generate a matrix multiplication "
                                "with given parameters. Matrix C and B (Trans) do not match")
      else:
        if self.mat_c.get_actual_num_cols() != self.mat_b.get_actual_num_cols():
          raise GenerationError("Cannot generate a matrix multiplication "
                                "with given parameters. Matrix C and B (NoTrans) do not match")

      # the contraction length of A and B can be different
      # due to the fact that matrices in a matrix chain can be aligned
      # in a slightly different way e.g., see yateto
      if self._strict_match:
        # check whether A and B match each other
        if self.trans_a:
          if self.trans_b:
            if self.mat_a.get_actual_num_rows() != self.mat_b.get_actual_num_cols():
              raise GenerationError("Cannot generate a matrix multiplication with given parameters. "
                                    "Matrix A (Trans) and B (Trans) do not match")
          else:
            if self.mat_a.get_actual_num_rows() != self.mat_b.get_actual_num_rows():
              raise GenerationError("Cannot generate a matrix multiplication with given parameters. "
                                    "Matrix A (Trans) and B (NoTrans) do not match")
        else:
          if self.trans_b:
            if self.mat_a.get_actual_num_cols() != self.mat_b.get_actual_num_cols():
              raise GenerationError("Cannot generate a matrix multiplication with given parameters. "
                                    "Matrix A (NoTrans) and B (Trans) do not match")
          else:
            if self.mat_a.get_actual_num_cols() != self.mat_b.get_actual_num_rows():
              raise GenerationError("Cannot generate a matrix multiplication with given parameters. "
                                    "Matrix A (NoTrans) and B (NoTrans) do not match")

    except GenerationError as err:
      print(self.mat_a.gen_descr())
      print(self.mat_b.gen_descr())
      print(self.mat_c.gen_descr())
      raise err

  def compute_flops(self):
    flops = (2 * self._k - 1) * self._m * self._n
    if self.beta != 0:
      flops += self._m * self._n
    return flops
