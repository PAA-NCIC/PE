set term pngcairo font "Times-New-Roman,20" size 800,600
set output "./png/bandwidth_ptrchase_random_sequential.png"

set xlabel "size of region (bytes)"
set ylabel "bandwidth (GB/sec)"
set title "list traveral vs random access vs seqential access"

set xrange [512:]
set logscale x 10
set xtics 1e3, 10
set format x "10^{%L}"
set key right

plot \
'./data/mem_local.dat' using 1:3 lw 1 with linespoints title "ptrchase", \
'./data/mem_random_without_ptrchase.dat' using 1:3 lw 1 with linespoints title "random", \
'./data/mem_seqential_without_ptrchase.dat' using 1:3 lw 1 with linespoints title "seqential", \

