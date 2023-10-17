set term pngcairo font "Times-New-Roman,20" size 1200,600
set output "./png/memory_bandwidth_various.png"

set xlabel "size of region (bytes)"
set ylabel "bandwidth (GB/sec)"
set title "summary of various access pattern"

set xrange [134217728:1073741824]
set logscale x 10
set xtics 1e9, 10
set format x "10^{%L}"
set key outside right center

plot \
'./data/mem_address_ordered_travel.dat' using 1:3 lw 1 with linespoints title "ptrchase(sorted)", \
'./data/mem_local.dat' using 1:3 lw 1 with linespoints title "ptrchase", \
'./data/mem_random_without_ptrchase.dat' using 1:3 lw 1 with linespoints title "random", \
'./data/mem_seqential_without_ptrchase.dat' using 1:3 lw 1 with linespoints title "sequential", \
'./data/mem_prefetch_10.dat' using 1:3 lw 1 with linespoints title "ptrchase(prefetch)", \
'./data/mem_10.dat' using 1:3 lw 1 with linespoints title "ptrchase(x10)", \

