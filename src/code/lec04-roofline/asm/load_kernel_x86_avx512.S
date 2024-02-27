.globl load_kernel_x86_avx512
.globl load_kernel_x86_avx512

load_kernel_x86_avx512:
    push %rbp
	xorl %eax, %eax
    
    #first parameter , len is in rdi
    #second parameter, src is in rsi
    #shrq   $4, %rdi

.READ_LOOP:
    addq      $256, %rsi
    vmovdqa64 -256(%rsi), %zmm0
    vmovdqa64 -192(%rsi), %zmm1
    vmovdqa64 -128(%rsi), %zmm2
    vmovdqa64 -64(%rsi),  %zmm3
    addq    $256, %rax
    cmpq    %rax, %rdi  
    jne .READ_LOOP

    pop %rbp
    ret

