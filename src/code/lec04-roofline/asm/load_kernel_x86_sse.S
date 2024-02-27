.globl load_kernel_x86_sse
.globl load_kernel_x86_sse

load_kernel_x86_sse:
    push %rbp
	xorl %eax, %eax
    
    #first parameter , len is in rdi
    #second parameter, src is in rsi
    #shrq   $4, %rdi

.READ_LOOP:
    addq      $64, %rsi
    movsd -64(%rsi), %xmm0
    movsd -48(%rsi), %xmm1
    movsd -32(%rsi), %xmm2
    movsd -16(%rsi), %xmm3
    addq    $64, %rax
    cmpq    %rax, %rdi  
    jne .READ_LOOP

    pop %rbp
    ret

