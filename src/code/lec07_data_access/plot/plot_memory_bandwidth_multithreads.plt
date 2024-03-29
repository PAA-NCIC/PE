set term pngcairo font "Times-New-Roman,20" size 1200,600
set output "./png/memory_bandwidth_multithreads.png"

set xlabel "size of region (bytes)"
set ylabel "bandwidth (GB/sec)"
set title "bandwidth with a number of threads"

set xrange [134217728:]
set logscale x 10
set xtics 1e9, 10
set format x "10^{%L}"
set key outside right center

plot \
'./data/mem_thread1_chain1.dat' using 1:3 lw 1 with linespoints title "1 chains, 1 threads", \
'./data/mem_thread1_chain10.dat' using 1:3 lw 1 with linespoints title "10 chains, 1 threads", \
'./data/mem_thread4_chain1.dat' using 1:3 lw 1 with linespoints title "1 chains, 4 threads", \
'./data/mem_thread4_chain10.dat' using 1:3 lw 1 with linespoints title "10 chains, 4 threads", \
'./data/mem_thread8_chain1.dat' using 1:3 lw 1 with linespoints title "1 chains, 8 threads", \
'./data/mem_thread8_chain10.dat' using 1:3 lw 1 with linespoints title "10 chains, 8 threads", \
'./data/mem_thread16_chain1.dat' using 1:3 lw 1 with linespoints title "1 chains, 16 threads", \
'./data/mem_thread16_chain10.dat' using 1:3 lw 1 with linespoints title "10 chains, 16 threads"

