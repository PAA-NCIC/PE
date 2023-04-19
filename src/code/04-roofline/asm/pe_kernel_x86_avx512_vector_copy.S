.globl pe_kernel_x86_avx512_vector_copy

pe_kernel_x86_avx512_vector_copy:
    vpxorq %zmm0, %zmm0, %zmm0
    vpxorq %zmm1, %zmm1, %zmm1
    vpxorq %zmm2, %zmm2, %zmm2
    vpxorq %zmm3, %zmm3, %zmm3
    vpxorq %zmm4, %zmm4, %zmm4
    vpxorq %zmm5, %zmm5, %zmm5
    vpxorq %zmm6, %zmm6, %zmm6
    vpxorq %zmm7, %zmm7, %zmm7
    vpxorq %zmm8, %zmm8, %zmm8
    vpxorq %zmm9, %zmm9, %zmm9
.pe.x86.avx512.vector.copy.L1:
    vfmadd231pd %zmm0, %zmm0, %zmm0
    vfmadd231pd %zmm1, %zmm1, %zmm1
    vfmadd231pd %zmm2, %zmm2, %zmm2
    vfmadd231pd %zmm3, %zmm3, %zmm3
    vfmadd231pd %zmm4, %zmm4, %zmm4
    vfmadd231pd %zmm5, %zmm5, %zmm5
    vfmadd231pd %zmm6, %zmm6, %zmm6
    vfmadd231pd %zmm7, %zmm7, %zmm7
    vfmadd231pd %zmm8, %zmm8, %zmm8
    vfmadd231pd %zmm9, %zmm9, %zmm9
    sub $0x1, %rdi
    jne .pe.x86.avx512.vector.copy.L1
    ret

