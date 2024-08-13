#ifndef _BST_H_
#define _BST_H_

typedef struct _Node Node;

Node * insertNode(Node * root, int value);

Node * deleteNode(Node * root, int value);

void printSubtree(Node * N);

int countLeaves(Node * N);

Node * deleteSubtree(Node * root, int value);

int depth (Node * R, Node * N); 

#endif
