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
'./data/cycle_address_ordered_travel.dat' using 1:(64/$2*2.9*1e9/1024/1024/1024) lw 1 with linespoints title "ptrchase(sorted)", \
'./data/cycle_local.dat' using 1:(64/$2*2.9*1e9/1024/1024/1024) lw 1 with linespoints title "ptrchase", \
'./data/cycle_random_without_ptrchase.dat' using 1:(64/$2*2.9*1e9/1024/1024/1024) lw 1 with linespoints title "random", \
'./data/cycle_seqential_without_ptrchase.dat' using 1:(64/$2*2.9*1e9/1024/1024/1024) lw 1 with linespoints title "sequential", \
'./data/cycle_prefetch_10.dat' using 1:(64/$2*2.9*1e9/1024/1024/1024) lw 1 with linespoints title "ptrchase(prefetch)", \
'./data/cycle_10.dat' using 1:(64/$2*2.9*1e9/1024/1024/1024) lw 1 with linespoints title "ptrchase(x10)", \

