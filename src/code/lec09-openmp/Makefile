
all: hello share_private schedule_example task_example reduction_example single_example race_condition deadlock loop_overhead  histogram numa_test page_migration  gauss_seidel

hello: hello.cpp
	g++ hello.cpp -fopenmp -o hello
share_private: share_private.cpp
	g++ share_private.cpp -fopenmp -o share_private
schedule_example: schedule.cpp
	g++ schedule.cpp -fopenmp -o schedule_example
single_example: single.cpp
	g++ single.cpp -fopenmp -o single_example
task_example: task.cpp
	g++ task.cpp -fopenmp -o task_example
reduction_example: reduction.cpp
	g++ reduction.cpp -fopenmp -o reduction_example
race_condition_v2: race_condition_v2.cpp
	g++ race_condition.cpp -fopenmp -o race_condition
deadlock: deadlock.cpp
	g++ deadlock.cpp -fopenmp -o deadlock
loop_overhead: loop_overhead.cpp
	g++ loop_overhead.cpp -fopenmp -o loop_overhead
histogram: histogram.cpp
	g++ histogram.cpp -fopenmp -o histogram
numa_test: numa_test.cpp
	g++ numa_test.cpp -fopenmp -o numa_test
page_migration: page_migration.cpp
	g++ page_migration.cpp -g -fopenmp -o page_migration
gauss_seidel: gauss_seidel.cpp
	g++ gauss_seidel.cpp -fopenmp -o gauss_seidel


clean:
	rm hello share_private schedule_example task_example reduction_example single_example race_condition deadlock loop_overhead  histogram numa_test page_migration  gauss_seidel -rf
