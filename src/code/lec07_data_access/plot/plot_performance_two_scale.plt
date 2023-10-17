set term pngcairo font "Times-New-Roman,20" size 700,600
set output "./png/performance_two_scale.png"

set xlabel "loop length N"
set ylabel "performance [MF/s]"

set xrange [256:134217728]
set logscale x 10
set xtics 1e3, 10
set format x "10^{%L}"
set key right

plot \
'./data/two_scale_mflops_naive.dat' using 1:2 with linespoints lw 1 title "no fusion", \
'./data/two_scale_mflops_fused.dat' using 1:2 with linespoints lw 1 title "loop fusion", \