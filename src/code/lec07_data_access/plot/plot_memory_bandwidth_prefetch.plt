set term pngcairo font "Times-New-Roman,20" size 800,600
set output "./png/menory_bandwidth_prefetch.png"

set xlabel "size of region (bytes)"
set ylabel "bandwidth (GB/sec)"
set title "bandwidth w/ and w/o prefetch"

set xrange [134217728:1073741824]
set logscale x 10
set xtics 10
set format x "10^{%L}"
set key right center

plot \
'./data/cycle_prefetch_0.dat' using 1:(64/$2*2.9*1e9/1024/1024/1024) lw 1 with linespoints title "prefetch=0", \
'./data/cycle_prefetch_10.dat' using 1:(64/$2*2.9*1e9/1024/1024/1024) lw 1 with linespoints title "prefetch=10",

