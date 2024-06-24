set term pngcairo font "Times-New-Roman,20" size 800,600
set output "./png/bandwidth_multichain.png"

set xlabel "size of region (bytes)"
set ylabel "bandwidth (GB/sec)"
set title "bandwidth with a number of chains"

set xrange [512:]
set logscale x 10
set xtics 1e3, 10
set format x "10^{%L}"
set key right

plot \
'./data/mem_local.dat' using 1:3 lw 1 with linespoints title "1 chains", \
'./data/mem_02.dat' using 1:3 lw 1 with linespoints title "2 chains", \
'./data/mem_04.dat' using 1:3 lw 1 with linespoints title "4 chains", \
'./data/mem_06.dat' using 1:3 lw 1 with linespoints title "6 chains", \
'./data/mem_08.dat' using 1:3 lw 1 with linespoints title "8 chains", \
'./data/mem_10.dat' using 1:3 lw 1 with linespoints title "10 chains", \
'./data/mem_12.dat' using 1:3 lw 1 with linespoints title "12 chains", \
# './data/mem_14.dat' using 1:3 lw 1 with linespoints title "14 chains", \
# './data/mem_16.dat' using 1:3 lw 1 with linespoints title "16 chains", \


