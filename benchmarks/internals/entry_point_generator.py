from .aux import Aux


class EnryPointGenerator:
  def __init__(self, benchmarks_names, benchmarks_src, config):
    self._benchmarks_names = benchmarks_names
    self._benchmarks_src = benchmarks_src
    self._config = config
    self._template = None

  def generate(self):
    self._template = Aux.read_template(template_name='entry_point.tmpl')
    self._adjust_template()

    src = self._template.render(benchmarks_names=self._benchmarks_names,
                                benchmarks_src=self._benchmarks_src,
                                config=self._config)
    return src

  def _adjust_template(self):
    def list_to_arg_string(args):
      decorated_args = [f'\"{arg}\"' for arg in args]
      return ','.join(decorated_args)

    # register callbacks
    self._template.globals['list_to_arg_string'] = list_to_arg_string
