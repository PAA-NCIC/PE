set term pngcairo font "Times-New-Roman,20" size 800,600
set output "./png/bandwidth_ptrchase_random_sequential.png"

set xlabel "size of region (bytes)"
set ylabel "bandwidth (GB/sec)"
set title "list traveral vs random access vs seqential access"

set xrange [4096:1073741824]
set logscale x 10
set xtics 1e4, 10
set format x "10^{%L}"
set key right

plot \
'./data/cycle_local.dat' using 1:(64/$2*2.9*1e9/1024/1024/1024) lw 1 with linespoints title "ptrchase", \
'./data/cycle_random_without_ptrchase.dat' using 1:(64/$2*2.9*1e9/1024/1024/1024) lw 1 with linespoints title "random", \
'./data/cycle_seqential_without_ptrchase.dat' using 1:(64/$2*2.9*1e9/1024/1024/1024) lw 1 with linespoints title "seqential", 

