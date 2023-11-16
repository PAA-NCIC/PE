#!/bin/bash
 #echo "nthreads,variables,flops,cycles,fmas,cycles/iter" > output
./axpy nthreads | tee tmp
echo "nthreads,variables,flops,cycles,fmas,cycles/iter" > output
lines=`wc -l  tmp | cut -d " " -f 1`
for ((line=1; line<=${lines};line+=6))
do
   let "begin=${line}"
   let "end=${line}+5"
   cut -d ":" -f 2 tmp | sed  -n "${begin},${end}p" | awk '{i=1;while(i <= NF){col[i]=col[i] $i ",";i=i+1}} END {i=1;while(i<=NF){print col[i];i=i+1}}' >> output
done



 
