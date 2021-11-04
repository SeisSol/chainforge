from chainforge.common import DenseMatrix
from chainforge.common import Context
from chainforge.common.aux import generate_tmp_matrix
from chainforge.common import GemmDescr, FloatingPointType, Addressing
from chainforge.backend.generator import Generator


# Q += (A x B^T) x B

mat_q = DenseMatrix(num_rows=56,
                    num_cols=56,
                    addressing=Addressing.PTR_BASED,
                    bbox=[0, 0, 20, 9])

mat_a = DenseMatrix(num_rows=56,
                    num_cols=56,
                    addressing=Addressing.NONE,
                    bbox=[0, 0, 20, 9])

mat_b = DenseMatrix(num_rows=56,
                    num_cols=56,
                    addressing=Addressing.STRIDED,
                    bbox=[0, 0, 20, 9])

tmp1 = generate_tmp_matrix(mat_a, mat_b, trans_op1=False, trans_op2=True)

gemm_list = [GemmDescr(trans_a=False,
                       trans_b=True,
                       a=mat_a, b=mat_b, c=tmp1),
             GemmDescr(trans_a=False,
                       trans_b=False,
                       a=tmp1, b=mat_b, c=mat_q)]

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