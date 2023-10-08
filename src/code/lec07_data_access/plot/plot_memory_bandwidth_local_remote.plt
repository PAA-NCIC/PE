set term pngcairo font "Times-New-Roman,20" size 800,600
set output "./png/memory_bandwidth_local_remote.png"

set xlabel "size of region (bytes)"
set ylabel "bandwidth (GB/sec)"
set title "bandwidth of random list traveral"

set xrange [134217728:1073741824]
set logscale x 10
set xtics 1e9, 10
set format x "10^{%L}"
set key right

plot \
'./data/cycle_local.dat' using 1:(64/$2*2.9*1e9/1024/1024/1024) lw 1 with linespoints title "local", \
'./data/cycle_remote.dat' using 1:(64/$2*2.9*1e9/1024/1024/1024) lw 1 with linespoints title "remote",

