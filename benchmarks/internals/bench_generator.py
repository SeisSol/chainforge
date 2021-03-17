from .gpu_api import GpuAPI
from .aux import Aux
from chainforge.common import Addressing


class BenchGenerator:
  def __init__(self, test_name, symbol_table, gemm_list, config, call_site, arch):
    self._test_name = test_name
    self._symbol_table = symbol_table
    self._gemm_list = gemm_list
    self._template = None
    self._config = config
    self._arch = arch
    self._call_site = call_site

  def generate(self):
    self._prune_symbol_table()
    self._template = Aux.read_template(template_name='bench.tmpl')
    self._adjust_template()
    cpu_matrices = self._symbol_table.get_all_matrices()
    cpu_matrices = self._decorate_matrices(cpu_matrices)

    gpu_matrices = self._symbol_table.get_matrices(0)
    gpu_matrices = self._decorate_matrices(gpu_matrices)
    res_matrix = self._gemm_list[-1].mat_c
    res_matrix = {'name': res_matrix.alias,
                  'attr': res_matrix}

    src = self._template.render(test_name=self._test_name,
                                cpu_matrices=cpu_matrices,
                                gpu_matrices=gpu_matrices,
                                res_matrix=res_matrix,
                                gemm_list=self._gemm_list,
                                call_site=self._call_site,
                                flops=self._compute_flops(),
                                config=self._config)
    return src

  def _decorate_matrices(self, matrices):
    matrix_descr = []
    for matrix in matrices:
      name, attr = matrix
      matrix_descr.append({'name': name,
                           'attr': attr.descr})
    return matrix_descr

  def _adjust_template(self):
    def is_batch(addressing):
      return False if addressing == Addressing.NONE else True

    def trans2str(is_trans):
      return 'LayoutType::Trans' if is_trans else 'LayoutType::NoTrans'

    # register callbacks
    self._template.globals['api'] = GpuAPI.get(self._arch)
    self._template.globals['is_batch'] = is_batch
    self._template.globals['trans2str'] = trans2str

  def _prune_symbol_table(self):
    used_variables = self._get_used_variable_names()
    all_variables = self._get_all_variable_names()
    unused_variables = all_variables - used_variables

    for variable in unused_variables:
      self._symbol_table.force_remove(variable)

  def _get_used_variable_names(self):
    used_variables = set()
    for gemm in self._gemm_list:
      used_variables.add(gemm.mat_a.alias)
      used_variables.add(gemm.mat_b.alias)
      used_variables.add(gemm.mat_c.alias)
    return used_variables

  def _get_all_variable_names(self):
    variables = set()
    for scope in self._symbol_table.scopes:
      for var in scope.vars.keys():
        variables.add(var)
    return variables

  def _compute_flops(self):
    flops = 0
    for gemm in self._gemm_list:
      flops += gemm.compute_flops()
    return flops
