set term pngcairo font "Times-New-Roman,20" size 800,600
set output "./png/L2_miss_dgemm.png"

set xlabel "M=N=K"
set ylabel "L2 Miss Rate"

set xrange [128:4096]
set key right

plot \
'./data/dgemm_naive.dat' using 1:4 with linespoints lw 1 title "no blocking", \
'./data/dgemm_recursively_blocked_32.dat' using 1:4 with linespoints lw 1 title "recursive blocking", \