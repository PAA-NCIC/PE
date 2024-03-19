set term pngcairo font "Times-New-Roman,20" size 800,600
set output "./png/latency_local_small_repeat.png"

set xlabel "size of region (bytes)"
set ylabel "latency per load (CPU cycles)"
set title "latency per load in a random list traveral"

set xrange [256:]
set logscale x 10
set xtics 1e3, 10
set format x "10^{%L}"
set key left

plot \
'./data/mem_local_small_repeat.dat' lw 1 with linespoints title "small repeat count", \
'./data/mem_local.dat' lw 1 with linespoints title "large repeat count"
