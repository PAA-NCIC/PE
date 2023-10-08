set term pngcairo font "Times-New-Roman,20" size 800,600
set output "./png/performance_dmvm_unroll.png"

set xlabel "number of rows R"
set ylabel "Performance [MF/s]"

set xrange [256:524288]
set logscale x 10
set xtics 1e3, 10
set format x "10^{%L}"
set key right

plot \
'./data/dmvm_mflops_naive.dat' with linespoints lw 1 title "no unroll", \
'./data/dmvm_mflops_unroll2.dat' with linespoints lw 1 title "unroll(2)", \


