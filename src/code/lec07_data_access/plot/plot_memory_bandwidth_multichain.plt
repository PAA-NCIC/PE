set term pngcairo font "Times-New-Roman,20" size 800,500
set output "./png/memory_bandwidth_multichain.png"

set xlabel "size of region (bytes)"
set ylabel "bandwidth (GB/sec)"
set title "bandwidth with a number of chains"

set xrange [134217728:1073741824]
set logscale x 10
set xtics 1e9, 10
set format x "10^{%L}"
set key outside right center

plot \
'./data/cycle_local.dat' using 1:(64/$2*2.9*1e9/1024/1024/1024) lw 1 with linespoints title "1 chains", \
'./data/cycle_02.dat' using 1:(64/$2*2.9*1e9/1024/1024/1024) lw 1 with linespoints title "2 chains", \
'./data/cycle_04.dat' using 1:(64/$2*2.9*1e9/1024/1024/1024) lw 1 with linespoints title "4 chains", \
'./data/cycle_06.dat' using 1:(64/$2*2.9*1e9/1024/1024/1024) lw 1 with linespoints title "6 chains", \
'./data/cycle_08.dat' using 1:(64/$2*2.9*1e9/1024/1024/1024) lw 1 with linespoints title "8 chains", \
'./data/cycle_10.dat' using 1:(64/$2*2.9*1e9/1024/1024/1024) lw 1 with linespoints title "10 chains", \
'./data/cycle_12.dat' using 1:(64/$2*2.9*1e9/1024/1024/1024) lw 1 with linespoints title "12 chains", \
'./data/cycle_14.dat' using 1:(64/$2*2.9*1e9/1024/1024/1024) lw 1 with linespoints title "14 chains", \
'./data/cycle_16.dat' using 1:(64/$2*2.9*1e9/1024/1024/1024) lw 1 with linespoints title "16 chains", \

