set term pngcairo font "Times-New-Roman,20" size 1200,400
set output "./png/data_traffic_dmvm_tiling.png"

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
'./data/dmvm_data_traffic_naive.dat' using 1:2 with lines lw 1 lt 1 dt 3 notitle, \
'./data/dmvm_data_traffic_naive.dat' using 1:3 with lines lw 1 lt 2 dt 3 notitle, \
'./data/dmvm_data_traffic_naive.dat' using 1:4 with lines lw 1 lt 3 dt 3 notitle, \
'./data/dmvm_data_traffic_tiling2048.dat' using 1:2 with lines lw 2 lt 1 dt 2 title "B_c^{L2}", \
'./data/dmvm_data_traffic_tiling2048.dat' using 1:3 with lines lw 2 lt 2 dt 4 title "B_c^{L3}", \
'./data/dmvm_data_traffic_tiling2048.dat' using 1:4 with lines lw 2 lt 3 dt 5 title "B_c^{Mem}", \

