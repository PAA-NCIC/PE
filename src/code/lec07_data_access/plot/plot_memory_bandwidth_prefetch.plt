set term pngcairo font "Times-New-Roman,20" size 1000,600
set output "./png/memory_bandwidth_prefetch.png"

set xlabel "size of region (bytes)"
set ylabel "bandwidth (GB/sec)"
set title "bandwidth w/ and w/o prefetch"

set xrange [134217728:]
set logscale x 10
set xtics 1e9, 10
set format x "10^{%L}"
set key outside right center

plot \
'./data/mem_prefetch_0.dat' using 1:3 lw 1 with linespoints title "prefetch=0", \
'./data/mem_prefetch_10.dat' using 1:3 lw 1 with linespoints title "prefetch=10", \
'./data/mem_prefetch_20.dat' using 1:3 lw 1 with linespoints title "prefetch=20", \
'./data/mem_prefetch_40.dat' using 1:3 lw 1 with linespoints title "prefetch=40", \
'./data/mem_prefetch_80.dat' using 1:3 lw 1 with linespoints title "prefetch=80", \
'./data/mem_prefetch_120.dat' using 1:3 lw 1 with linespoints title "prefetch=120", \
# './data/mem_prefetch_160.dat' using 1:3 lw 1 with linespoints title "prefetch=160", \

