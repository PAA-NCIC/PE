
all: hello share_private schedule_example task_example reduction_example single_example

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

clean:
	rm hello share_private schedule_example task_example reduction_example single_example
