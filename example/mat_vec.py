from chainforge.common import DenseMatrix
from chainforge.common import Context
from chainforge.common.aux import generate_tmp_matrix
from chainforge.common import GemmDescr, FloatingPointType, Addressing
from chainforge.backend.generator import Generator


# C += A x b

vec_c = DenseMatrix(num_rows=56,
                    num_cols=1,
                    addressing=Addressing.STRIDED,
                    bbox=[0, 0, 56, 1])

mat_a = DenseMatrix(num_rows=56,
                    num_cols=9,
                    addressing=Addressing.STRIDED,
                    bbox=[0, 0, 56, 9])

vec_b = DenseMatrix(num_rows=56,
                    num_cols=1,
                    addressing=Addressing.STRIDED,
                    bbox=[0, 0, 9, 1])


gemm_list = [GemmDescr(trans_a=False,
                       trans_b=False,
                       a=mat_a, b=vec_b, c=vec_c)]

context = Context(arch='sm_60',
                  backend='cuda',
                  fp_type=FloatingPointType.FLOAT)

generator = Generator(gemm_list, context)
generator.generate()

print(generator.get_launcher())
print()
print(generator.get_header())
print()
print(generator.get_kernel())