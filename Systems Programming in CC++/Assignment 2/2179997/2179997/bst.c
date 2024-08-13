#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include <unistd.h>
#include "bst.h"
#include <stdbool.h>
#include <assert.h>

/*


Place for the BST functions from Exercise 1.

*/

/*
   Returns the parent of an either existing or hypotetical node with the given data
 */
Node * find_parent(Node * root, int data) {
    assert(root != NULL);
    assert(data != root->data);

    Node * next = data < root->data ? root->left : root->right;

    if (next == NULL || next->data == data)
        return root;
    else
        return find_parent(next, data);
}

/*
   Constructs a new node
 */
Node * mk_node(int data) {
    Node * node = (Node *) malloc(sizeof(Node));
    node->data = data;
    node->left = node->right = NULL;
    return node;
}

Node * insertNode(Node * root, int data) {
    if (root == NULL)
        return mk_node(data);

    if (data == root->data)
        return NULL;

    Node * parent = find_parent(root, data);
    Node * child = data < parent->data ? parent->left : parent->right;
    assert(child == NULL || child->data == data);

    if (child == NULL) {
        // data not found, then insert and return node
        child = mk_node(data);
        if (data < parent->data)
            parent->left = child;
        else
            parent->right = child;

        return child;
    } else {
        // data found, then return null
        return NULL;
    }
}


bool is_ordered(Node * root) {
    if (root == NULL)
        return true;
    if (root->left && root->left->data > root->data)
        return false;
    if (root->right && root->right->data < root->data)
        return false;
    return true;
}

Node * deleteNode(Node * root, int data) {
    assert(is_ordered(root));

    // empty tree
    if (root == NULL)
        return NULL;

    // find node with data 'data' and its parent node
    Node * parent, * node;
    if (root->data == data) {
        parent = NULL;
        node = root;
    } else {
        parent = find_parent(root, data);
        node = data < parent->data ? parent->left : parent->right;
    }
    assert(node == NULL || node->data == data);

    // data not found
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
        if (data < parent->data)
            parent->left = new_node;
        else
            parent->right = new_node;
    }

    assert(is_ordered(new_root));

    return new_root;
}

void printSubtree(Node * N) {
    if (N == NULL) {
        return;
    }

    printSubtree(N->left);
    printf("%d \n", N->data);
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
   whose root node has data 'data'
 */
Node * deleteSubtree(Node * root, int data) {
    assert(is_ordered(root));

    // empty tree
    if (root == NULL)
        return NULL;

    // entire tree
    if (root->data == data) {
        free_subtree(root);
        return NULL;
    }

    // free tree rooted in the left or right node and set the respective pointer to NULL
    Node * parent = find_parent(root, data);
    if (data < parent->data) {
        assert(parent->left == NULL || parent->left->data == data);
        free_subtree(parent->left);
        parent->left = NULL;
    } else {
        assert(parent->right == NULL || parent->right->data == data);
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

    int sub_depth = depth(R->data > N->data ? R->left : R->right, N);

    if (sub_depth >= 0)
        return sub_depth + 1;
    else
        return -1;
}

Node* freeSubtree(Node *N) {
    if (N == NULL)
    {
        free(N);
        return NULL;
    }

    N->left = freeSubtree(N->left);
    N->right = freeSubtree(N->right);
    free(N);

    return NULL;
}

int countNodes(Node *N) {
    if (N == NULL)
    {
        return  0;
    }
    else
    {
        return (countNodes(N->left) + 1 + countNodes(N->right));
    }
}


int sumSubtree(Node *N)
{

    // TODO: Implement this function
    if (N == NULL)
    {
        return 0;
    }
    else
    {
        return sumSubtree(N->left) + N->data + (sumSubtree(N->right));
    }




}

struct linkedListNode
{
    int data;
    struct linkedListNode* next;
};

struct linkedList
{
    struct linkedListNode *head;
};

typedef struct linkedList linkedList;
typedef struct linkedListNode linkedListNode;

linkedListNode * allocateMemory(int data)
{
    linkedListNode *new = (linkedListNode *) malloc(sizeof(struct linkedListNode));
    if (new == NULL) {
        printf("Error: could not assign memory");
        return NULL;
    }
    new->data = data;
    new->next = NULL;
    return new;
}

linkedList * addItem(linkedList * list, int data) {
    if (list != NULL) {
        linkedListNode *current = list->head;
        //go to end of list
        if (current != NULL) {
            while (current->next != NULL) {
                current = current->next;
            }
        }


        //allocate memory
        linkedListNode * new = allocateMemory(data);


        //now append the element

        current->next = new;
        return list;
    }
    else
    {
        linkedList*  returnList = (linkedList*) malloc(sizeof(struct linkedList));
        returnList -> head = allocateMemory(data);
        return returnList;
    }

}

linkedList * makeList(Node* root, linkedList * list)
{
    if(root == NULL)
    {
        return list;
    }
    else
    {
        //pass left half of the tree                                  32
        //   3
        list = makeList(root->left, list);//     7
        //pass the root                                      //       9
        list = addItem(list, root->data);
        //pass the right root
        list = makeList(root->right,list);

        return list;
    }
}

void showLinkedList(linkedList * list)
{
    //checks if empty
    if(list == NULL)
    {
        return;
    }

    linkedListNode * current = list->head;
    while(current != NULL)
    {
        printf("%i ",current->data);
        current = current->next;
    }
}



linkedList ** halfList3(linkedList * list, int halfway, int size)
{
    linkedList * firstHalf = list;
    linkedList * secondHalf = (linkedList*) malloc(sizeof(struct linkedList));

    linkedListNode * currentNode = firstHalf->head;
    for(int i = 0; i< halfway - 1; i++)
    {
        currentNode = currentNode->next;
    }

    linkedList * midPoint = (linkedList*) malloc(sizeof(struct linkedList));
    midPoint->head = currentNode->next;


    currentNode->next = NULL;


    secondHalf->head = midPoint->head->next;
    midPoint->head->next = NULL;



    linkedList * listArray[3];
    listArray[0] = firstHalf;
    listArray[1] = midPoint;
    listArray[2] = secondHalf;

    linkedList * pointer = (linkedList *) &listArray;

    return (linkedList **) pointer;
}

Node* balance(linkedList *list, int size, Node * root, int firstNode) {
    if (list == NULL) {
        return NULL;  // 3 7 9 32
    } else {
        int newSize = div(size, 2).quot;
        if (size > 2) {

            linkedList *listArray[3];
            linkedList (**pointer) = (linkedList **) &listArray;

            pointer = halfList3(list, newSize, size);

            linkedList *first = (*pointer);
            linkedList *mid = (*(pointer + 1));
            linkedList *second = (*(pointer + 2));


            Node * tempNode;
            if (firstNode == 0)
            {
                root = insertNode(NULL, mid->head->data);
                firstNode ++;
            }
            else
            {
                tempNode = insertNode(root, mid->head->data);
            }
            root = balance(first, newSize, root, firstNode);
            root = balance(second, size - newSize - 1, root, firstNode);
            free(mid);
            return root;
        }
        else if (size == 2)
        {
            if (list->head->data < list->head->next->data)
            {
                Node * tempNode;
                if (firstNode == 0)
                {
                    root = insertNode(root,list->head->next->data);
                    firstNode ++;
                }
                else
                {
                    tempNode = insertNode(root,list->head->next->data);
                }

                tempNode = insertNode(root,list->head->data);
                free(list->head->next);
                free(list->head);
                free(list);
                return root;
            }
            else
            {
                Node * tempNode;

                if (firstNode == 0)
                {
                    root = insertNode(root,list->head->data);
                    firstNode ++;
                }
                else
                {
                    tempNode = insertNode(root,list->head->data);
                }

                tempNode = insertNode(root,list->head->next->data);
                free(list->head->next);
                free(list->head);
                free(list);
                return root;
            }
        }
        else
        {
            Node * tempNode;
            if (firstNode == 0)
            {
                root = insertNode(root,list->head->data);
                firstNode ++;
            }
            else
            {
                tempNode = insertNode(root,list->head->data);
            }
            free(list->head);
            free(list);
            return root;
        }



//        balance(*(listArray), div(newSize, 2).quot);
    }
}

// This functions converts an unbalanced BST to a balanced BST
Node *balanceTree(Node *root) {
    // TODO: Implement this function

    int size = countNodes(root);
    linkedList *list = makeList(root, NULL);
    Node * newRoot = NULL;
    return balance(list, size, newRoot, 0);
}





