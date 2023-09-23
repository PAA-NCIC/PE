#include <iostream>
#include <unistd.h>
#include <fstream>
#include <string>
using namespace std;

void trace_io_test(string input_file, string output_file) {
  string line;
  ifstream infile (input_file);
  ofstream outfile (output_file);
  if (infile.is_open() && outfile.is_open())
  {
    while ( getline (infile,line) )
    {
      outfile << line << '\n';
    }
    infile.close();
    outfile.close();
  }
}

int test1_2(){
	usleep(50);
  int sum = 0;
	for(int i = 0; i < 10000; i++) 
    sum += i;
  return sum;
}
int test1(){
  int sum = 0;
	for(int i = 0; i < 500; i++) 
    sum += test1_2();
  return sum;
}

int test2_2(){
	usleep(50);
  int sum = 0;
	for(int i = 0; i < 100000; i++) 
    sum += (i*i);
  return sum;
}
int test2(){
  int sum = 0;
	for(int i = 0; i < 10; i++) 
    sum += test2_2();
  return sum;
}

int test3_1(){
  int sum = 0;
	for(int i = 0; i < 10; i++) 
    sum+=i;
  return sum;
}

int test3_2(){
  int sum = 0;
	for(int i = 0; i < 100000; i++) 
    sum += i;
  return sum;
}

int test3(){
	for(int i = 0; i < 1000; i++) {
    test3_1();
    test3_2();
  }
  return 0;
}

int main(){
	trace_io_test("intput_file", "output_file");
	test1();
	test2();
	test3();
}
