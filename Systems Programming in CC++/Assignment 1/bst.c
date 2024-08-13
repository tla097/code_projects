#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>

#include "bst.h"

struct _Node {
  int value;
  Node * left, * right;
};

/* 
   Returns the parent of an either existing or hypotetical node with the given value  
 */
Node * find_parent(Node * root, int value) {
  assert(root != NULL);
  assert(value != root->value);

  Node * next = value < root->value ? root->left : root->right;

  if (next == NULL || next->value == value)
    return root;
  else
    return find_parent(next, value);
}

/* 
   Constructs a new node
 */
Node * mk_node(int value) {
  Node * node = (Node *) malloc(sizeof(Node));
  node->value = value;
  node->left = node->right = NULL;
  return node;
}

Node * insertNode(Node * root, int value) {
  if (root == NULL)
    return mk_node(value);

  if (value == root->value)
    return NULL;

  Node * parent = find_parent(root, value);
  Node * child = value < parent->value ? parent->left : parent->right;
  assert(child == NULL || child->value == value);

  if (child == NULL) {
    // value not found, then insert and return node   
    child = mk_node(value);
    if (value < parent->value)
      parent->left = child;
    else
      parent->right = child;

    return child;
  } else {
    // value found, then return null
    return NULL;
  } 
}

bool is_ordered(Node * root) {
  if (root == NULL)
    return true;
  if (root->left && root->left->value > root->value)
    return false;
  if (root->right && root->right->value < root->value)
    return false;
  return true;
}

Node * deleteNode(Node * root, int value) {
  assert(is_ordered(root));

  // empty tree
  if (root == NULL)
    return NULL;

  // find node with value 'value' and its parent node 
  Node * parent, * node;
  if (root->value == value) {
    parent = NULL;
    node = root;
  } else {
    parent = find_parent(root, value);
    node = value < parent->value ? parent->left : parent->right;
  }
  assert(node == NULL || node->value == value);

  // value not found
  if (node == NULL)
    return root;

  // re-establish consistency
  Node * new_node;
  if (node->left == NULL) {
    // node has only right child, then make right child the new node
    new_node = node->right;
  } else {
    // otherwise make right child the rightmost leaf of the subtree rooted in the left child
    // and make the left child the new node
    Node * rightmost = new_node = node->left;
    while (rightmost->right != NULL)
      rightmost = rightmost->right;
    rightmost->right = node->right;
  }

  free(node);

  Node * new_root;
  if (parent == NULL) {
    // if deleted node was root, then return new node as root 
    new_root = new_node;
  } else {
    // otherwise glue new node with parent and return old root
    new_root = root;
    if (value < parent->value) 
      parent->left = new_node;
    else 
      parent->right = new_node;
  }

  assert(is_ordered(new_root));

  return new_root;
}

void printSubtree(Node * N) {
  if (N == NULL) return;

  printSubtree(N->left);
  printf("%d \n", N->value);
  printSubtree(N->right);
}

int countLeaves(Node * N) {
  if (N == NULL)
    return 0;

  if (N->left == NULL && N->right == NULL)
    return 1;
  
  return countLeaves(N->left) + countLeaves(N->right);
}

/* 
   Frees the entire subtree rooted in 'root' (this includes the node 'root')
 */
void free_subtree(Node * root) {
  if (root == NULL)
    return;

  free_subtree(root->left);
  free_subtree(root->right);
  free(root);
}

/* 
   Deletes all nodes that belong to the subtree (of the tree of rooted in 'root') 
   whose root node has value 'value'
 */
Node * deleteSubtree(Node * root, int value) {
  assert(is_ordered(root));

  // empty tree
  if (root == NULL)
    return NULL;

  // entire tree
  if (root->value == value) {
    free_subtree(root);
    return NULL;
  }

  // free tree rooted in the left or right node and set the respective pointer to NULL 
  Node * parent = find_parent(root, value);
  if (value < parent->value) {
    assert(parent->left == NULL || parent->left->value == value);
    free_subtree(parent->left);
    parent->left = NULL;
  } else {
    assert(parent->right == NULL || parent->right->value == value);
    free_subtree(parent->right);
    parent->right = NULL;
  }

  return root;
}

/* 
   Compute the depth between root R and node N 

   Returns the number of edges between R and N if N belongs to the tree rooted in R,
   otherwise it returns -1
 */
int depth (Node * R, Node * N) {
  if (R == NULL || N == NULL)
    return -1;
  if (R == N)
    return 0;

  int sub_depth = depth(R->value > N->value ? R->left : R->right, N);

  if (sub_depth >= 0)
    return sub_depth + 1;
  else
    return -1;
}
