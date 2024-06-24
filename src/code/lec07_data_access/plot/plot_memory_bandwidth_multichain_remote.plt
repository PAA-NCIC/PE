set term pngcairo font "Times-New-Roman,20" size 800,500
set output "./png/memory_bandwidth_multichain_remote.png"

set xlabel "size of region (bytes)"
set ylabel "bandwidth (GB/sec)"
set title "bandwidth with a number of chains"

set xrange [134217728:]
set logscale x 10
set xtics 1e8, 10
set format x "10^{%L}"
set key outside right center

plot \
'./data/mem_remote.dat' using 1:3 lw 1 with linespoints title "1 chains", \
'./data/mem_02_remote.dat' using 1:3 lw 1 with linespoints title "2 chains", \
'./data/mem_04_remote.dat' using 1:3 lw 1 with linespoints title "4 chains", \
'./data/mem_06_remote.dat' using 1:3 lw 1 with linespoints title "6 chains", \
'./data/mem_08_remote.dat' using 1:3 lw 1 with linespoints title "8 chains", \
'./data/mem_10_remote.dat' using 1:3 lw 1 with linespoints title "10 chains", \
'./data/mem_12_remote.dat' using 1:3 lw 1 with linespoints title "12 chains", \
# './data/mem_14_remote.dat' using 1:3 lw 1 with linespoints title "14 chains", \
# './data/mem_16_remote.dat' using 1:3 lw 1 with linespoints title "16 chains", \

