set term pngcairo font "Times-New-Roman,20" size 600,600
set output "./png/data_traffic_dmvm_unroll2.png"

set xlabel "Numbe of rows R"
set ylabel "Data traffic [B/interation]"

set yrange [0:32]
set ytics 8
set xrange [256:524288]
set logscale x 10
set xtics 1e3,10
set format x "10^{%L}"
set key  top left

plot \
'./data/dmvm_data_traffic_unroll2.dat' using 1:2 with lines lw 2 title "B_c^{L2}", \
'./data/dmvm_data_traffic_unroll2.dat' using 1:3 with lines lw 2 dt 2 title "B_c^{L3}", \
'./data/dmvm_data_traffic_unroll2.dat' using 1:4 with lines lw 2 dt 3 title "B_c^{Mem}", \

