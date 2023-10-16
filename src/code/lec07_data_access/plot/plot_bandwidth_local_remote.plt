set term pngcairo font "Times-New-Roman,20" size 800,600
set output "./png/bandwidth_local_remote.png"

set xlabel "size of region (bytes)"
set ylabel "bandwidth (GB/sec)"
set title "bandwidth of random list traveral"

set xrange [256:1073741824]
set logscale x 10
set xtics 1e3, 10
set format x "10^{%L}"
set key right

plot \
'./data/mem_local.dat' using 1:3 lw 1 with linespoints title "local", \
'./data/mem_remote.dat' using 1:3 lw 1 with linespoints title "remote",

