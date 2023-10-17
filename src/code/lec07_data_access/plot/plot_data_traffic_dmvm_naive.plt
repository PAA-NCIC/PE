set term pngcairo font "Times-New-Roman,20" size 1200,400
set output "./png/data_traffic_dmvm_naive.png"

set xlabel "R"
set ylabel "Data traffic B/interation"

set yrange [0:32]
set ytics 8
set xrange [256:524288]
set logscale x 10
set xtics 1e3,10
set format x "10^{%L}"
set key  center center

plot \
'./data/dmvm_data_traffic_naive.dat' using 1:2 with lines lw 2 title "L2 <-> L1", \
'./data/dmvm_data_traffic_naive.dat' using 1:3 with lines lw 2 title "L3 <-> L2", \
'./data/dmvm_data_traffic_naive.dat' using 1:4 with lines lw 2 title "Memory <-> L3", \

