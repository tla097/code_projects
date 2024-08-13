#include <stdio.h>
#include <assert.h>

#include "bst.h"

int main() {
  Node *a, *b;
  a = insertNode(NULL, 42);
  b = deleteNode(a, 42);
  assert(b == NULL);

  a = insertNode(NULL, 1);
  b = insertNode(a, 2);

  printSubtree(a);
  int c = countLeaves(a);
  assert(c == 1);

  c = depth(a,b);
  assert(c == 1);

  a = deleteSubtree(a, 2);
  
  return 0;
}
