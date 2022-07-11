from chainforge.common import DenseMatrix
from chainforge.common import Context
from chainforge.common.aux import generate_tmp_matrix
from chainforge.common import GemmDescr, FloatingPointType, Addressing
from chainforge.backend.generator import Generator


# Q = (((A x B) x (C x B)) x D)
mat_q = DenseMatrix(num_rows=56,
                    num_cols=9,
                    addressing=Addressing.STRIDED,
                    bbox=[0, 0, 56, 9],)

mat_a = DenseMatrix(num_rows=56,
                    num_cols=56,
                    addressing=Addressing.STRIDED,
                    bbox=[0, 0, 56, 56])

mat_b = DenseMatrix(num_rows=56,
                    num_cols=9,
                    addressing=Addressing.STRIDED,
                    bbox=[0, 0, 56, 9])

mat_c = DenseMatrix(num_rows=56,
                    num_cols=56,
                    bbox=[0, 0, 56, 56],
                    addressing=Addressing.STRIDED)

mat_d = DenseMatrix(num_rows=9,
                    num_cols=9,
                    bbox=[0, 0, 9, 9],
                    addressing=Addressing.STRIDED)


tmp0 = generate_tmp_matrix(mat_a, mat_b)
tmp1 = generate_tmp_matrix(mat_c, mat_b)
tmp2 = generate_tmp_matrix(tmp0, tmp1)

gemm_list = [GemmDescr(trans_a=False,
                       trans_b=False,
                       a=mat_a, b=mat_b, c=tmp0),
             GemmDescr(trans_a=False,
                       trans_b=False,
                       a=mat_c, b=mat_b, c=tmp1),
             GemmDescr(trans_a=False, trans_b=False,
                       a=tmp0, b=tmp1, c=tmp2),
             GemmDescr(trans_a=False, trans_b=False,
                       a=tmp2, b=mat_d, c=mat_q,
                       alpha=1.0, beta=0.0),
             ]

context = Context(arch='sm_60',
                  backend='cuda',
                  fp_type=FloatingPointType.FLOAT)

generator = Generator(gemm_list, context)
generator.generate()

with_output = True
if with_output:
  print(generator.get_header())
  print(generator.default_generate_call_site())
  print()
  print(generator.get_launcher())
  print()
  print(generator.get_kernel())
