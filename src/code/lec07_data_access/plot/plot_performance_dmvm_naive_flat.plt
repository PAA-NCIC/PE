set term pngcairo font "Times-New-Roman,20" size 1200,400
set output "./png/performance_dmvm_naive_flat.png"

set xlabel "R"
set ylabel "MF/s"

set xrange [256:524288]
set logscale x 10
set xtics 1e3,10
set format x "10^{%L}"
set key right

plot \
'./data/dmvm_mflops_naive.dat' with linespoints lw 1 title "naive", \


