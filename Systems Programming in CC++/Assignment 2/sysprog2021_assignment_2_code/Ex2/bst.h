struct Node{
	int data;
	struct Node *left;
	struct Node *right;
};
typedef struct Node Node;

Node* insertNode(Node *root, int value);
Node* freeSubtree(Node *N);
int countNodes(Node *N);
Node* deleteNode(Node* root, int value);
void printSubtree(Node *N);
int sumSubtree(Node *N);
Node* balanceTree(Node* root);
