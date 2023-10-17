set term pngcairo font "Times-New-Roman,20" size 800,600
set output "./png/performance_vector_traids_avx.png"

set xlabel "N"
set ylabel "MF/s"

set xrange [32:134217728]
set logscale x 10
set xtics 1e2, 10
set format x "10^{%L}"
set key right

plot \
'./data/triads_mflops_naive.dat' with linespoints lw 1 title "scalar", \
'./data/triads_mflops_avx512.dat' with linespoints lw 1 title "AVX", \
'./data/triads_mflops_model.dat' with lines lw 1 dt 2 title "Simple Data Transfer Model", \

