set term pngcairo font "Times-New-Roman,20" size 800,600
set output "./png/bandwidth_address_oredered.png"

set xlabel "size of region (bytes)"
set ylabel "bandwidth (GB/sec)"
set title "bandwidth of random list traveral vs address oredered traveral"

set xrange [4096:1073741824]
set logscale x 10
set xtics 1e4, 10
set format x "10^{%L}"
set key right center

plot \
'./data/cycle_local.dat' using 1:(64/$2*2.9*1e9/1024/1024/1024) lw 1 with linespoints title "random list", \
'./data/cycle_address_ordered_travel.dat' using 1:(64/$2*2.9*1e9/1024/1024/1024) lw 1 with linespoints title "address oredered traveral",

