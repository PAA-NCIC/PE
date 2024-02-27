.globl load_kernel_x86_avx
.globl load_kernel_x86_avx

load_kernel_x86_avx:
    push %rbp
	xorl %eax, %eax
    
    #first parameter , len is in rdi
    #second parameter, src is in rsi
.READ_LOOP:
    addq      $128, %rsi
    vmovdqa64 -128(%rsi), %ymm0
    vmovdqa64 -96(%rsi), %ymm1
    vmovdqa64 -64(%rsi), %ymm2
    vmovdqa64 -32(%rsi),  %ymm3
    addq    $128, %rax
    cmpq    %rax, %rdi  
    jne .READ_LOOP

    pop %rbp
    ret

