#include <stdio.h>
#include <omp.h>
#include<iostream>
using namespace std;

typedef struct ListNode{
  struct ListNode *next;
  int data;
} ListNode;

void process_list(ListNode *head) {
  #pragma omp parallel
  {
    #pragma omp single
    {
      ListNode *p = head;
      while(p) {
        #pragma omp task 
        {p->data = p->data + 1;}
        p = p->next;
      }
    }
  }
}

void init_list(ListNode **head, int len) {
  *head = nullptr;
  ListNode *cur_ptr = nullptr;
  if(len > 0) {
    cout << "malloc head" << endl;
    ListNode *node = new ListNode;
    node->data = 0;
    node->next = nullptr;
    *head = cur_ptr = node;
    len--;
  }
  while(len > 0) {
    ListNode *node = new ListNode;
    node->data = 0;
    node->next = nullptr;
    cur_ptr->next = node;
    cur_ptr = node;
    len--;
  }
  
}

void print_list(ListNode *head) {
  ListNode *cur = head;
  cout << "List:\n";
  while(cur != nullptr) {
    cout << "  data: " << cur->data << "\n";
    cur = cur->next;
  }
}

void destroy_list(ListNode *head) {
  ListNode *cur = head;
  while(cur != nullptr) {
    ListNode *next = cur->next; 
    free(cur);
    cur = next;
  }
  head = nullptr;
}

int main(int argc, char* argv[]){

  ListNode *head = nullptr; 
  init_list(&head, 10);
  print_list(head);
  process_list(head);
  print_list(head);
  destroy_list(head);
  return 0;
}

