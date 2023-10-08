set term pngcairo font "Times-New-Roman,20" size 1200,600
set output "./png/memory_bandwidth_without_ptrchase_multithreads.png"

set xlabel "size of region (bytes)"
set ylabel "bandwidth (GB/sec)"
set title "bandwidth with various methods and number of threads"

set xrange [134217728:1073741824]
set logscale x 10
set xtics 1e9, 10
set format x "10^{%L}"
set key outside right center

plot \
'./data/cycle_random_without_ptrchase_thread1.dat' using 1:(64/$2*2.9*1e9/1024/1024/1024) lw 1 with linespoints title "random, 1 threads", \
'./data/cycle_seqential_without_ptrchase_thread1.dat' using 1:(64/$2*2.9*1e9/1024/1024/1024) lw 1 with linespoints title "seqential, 1 threads", \
'./data/cycle_random_without_ptrchase_thread8.dat' using 1:(64/$2*2.9*1e9/1024/1024/1024) lw 1 with linespoints title "random, 8 threads", \
'./data/cycle_seqential_without_ptrchase_thread8.dat' using 1:(64/$2*2.9*1e9/1024/1024/1024) lw 1 with linespoints title "seqential, 8 threads", \
'./data/cycle_random_without_ptrchase_thread12.dat' using 1:(64/$2*2.9*1e9/1024/1024/1024) lw 1 with linespoints title "random, 12 threads", \
'./data/cycle_seqential_without_ptrchase_thread12.dat' using 1:(64/$2*2.9*1e9/1024/1024/1024) lw 1 with linespoints title "seqential, 12 threads", \
'./data/cycle_random_without_ptrchase_thread16.dat' using 1:(64/$2*2.9*1e9/1024/1024/1024) lw 1 with linespoints title "random, 16 threads", \
'./data/cycle_seqential_without_ptrchase_thread16.dat' using 1:(64/$2*2.9*1e9/1024/1024/1024) lw 1 with linespoints title "seqential, 16 threads", \

