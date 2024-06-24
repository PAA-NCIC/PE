set term pngcairo font "Times-New-Roman,20" size 1200,600
set output "./png/memory_bandwidth_without_ptrchase_multithreads.png"

set xlabel "size of region (bytes)"
set ylabel "bandwidth (GB/sec)"
set title "bandwidth with various methods and number of threads"

set xrange [134217728:]
set logscale x 10
set xtics 1e9, 10
set format x "10^{%L}"
set key outside right center

plot \
'./data/mem_random_without_ptrchase_thread1.dat' using 1:3 lw 1 with linespoints title "random, 1 threads", \
'./data/mem_seqential_without_ptrchase_thread1.dat' using 1:3 lw 1 with linespoints title "seqential, 1 threads", \
'./data/mem_random_without_ptrchase_thread2.dat' using 1:3 lw 1 with linespoints title "random, 2 threads", \
'./data/mem_seqential_without_ptrchase_thread2.dat' using 1:3 lw 1 with linespoints title "seqential, 2 threads", \
'./data/mem_random_without_ptrchase_thread4.dat' using 1:3 lw 1 with linespoints title "random, 4 threads", \
'./data/mem_seqential_without_ptrchase_thread4.dat' using 1:3 lw 1 with linespoints title "seqential, 4 threads", \
'./data/mem_random_without_ptrchase_thread8.dat' using 1:3 lw 1 with linespoints title "random, 8 threads", \
'./data/mem_seqential_without_ptrchase_thread8.dat' using 1:3 lw 1 with linespoints title "seqential, 8 threads", \
'./data/mem_random_without_ptrchase_thread12.dat' using 1:3 lw 1 with linespoints title "random, 12 threads", \
'./data/mem_seqential_without_ptrchase_thread12.dat' using 1:3 lw 1 with linespoints title "seqential, 12 threads", \


