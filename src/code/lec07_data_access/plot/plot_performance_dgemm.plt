set term pngcairo font "Times-New-Roman,20" size 800,600
set output "./png/performance_dgemm.png"

set xlabel "M=N=K"
set ylabel "MF/s"

set xrange [128:4096]
set key right

plot \
'./data/dgemm_naive.dat' using 1:2 with linespoints lw 1 title "naive", \
'./data/dgemm_recursively_blocked_32.dat' using 1:2 with linespoints lw 1 title "recursively blocked 32", \