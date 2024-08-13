#include <stdbool.h>
#include <assert.h>

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_rwlock_t rwLock = PTHREAD_RWLOCK_INITIALIZER;

Node * makeNode(int data) {
    Node * node = (Node *) malloc(sizeof(Node));
    node->data = data;
    node->left = node->right = NULL;
    return node;
}

pthread_rwlock_t createNewLock()
{
    pthread_rwlock_t newLock = PTHREAD_RWLOCK_INITIALIZER;
    return newLock;
}



Node * findConcurrentParent(Node * root, int data) {
    assert(root != NULL);
    assert(data != root->data);

    Node * next = data < root->data ? root->left : root->right;

    if (next == NULL || next->data == data)
        return root;
    else
        return findConcurrentParent(next, data);
}




void insertNodeByRoot(int data) {
    if (root == NULL) {
        pthread_rwlock_unlock(&rwLock);
        pthread_rwlock_wrlock(&rwLock);
        root = makeNode(data);
        return;
    }

    if (data == root->data)
    {
        return;
    }

    Node * parent = findConcurrentParent(root, data);
    Node * child = data < parent->data ? parent->left : parent->right;
    assert(child == NULL || child->data == data);

    if (child == NULL) {
        pthread_rwlock_unlock(&rwLock);
        pthread_rwlock_wrlock(&rwLock);
        child = makeNode(data);
        if (data < parent->data)
            parent->left = child;
        else
            parent->right = child;
        return;
    } else {
        // data found, then return null
        return;
    }
}


bool isOrderedConcurrently(Node * root) {
    if (root == NULL)
        return true;
    if (root->left && root->left->data > root->data)
        return false;
    if (root->right && root->right->data < root->data)
        return false;
    return true;
}

void deleteNodeConcurrently(int data) {
    assert(isOrderedConcurrently(root));

    // empty tree
    if (root == NULL)
        return;

    // find node with data 'data' and its parent node
    Node * parent, * node;
    if (root->data == data)
    {
        parent = NULL;
        node = root;
    }
    else
    {
        parent = findConcurrentParent(root, data);
        node = data < parent->data ? parent->left : parent->right;
    }
    assert(node == NULL || node->data == data);

    // data not found
    if (node == NULL)
        return;

    // re-establish consistency
    Node * new_node;
    if (node->left == NULL)
    {

        // node has only right child, then make right child the new node
        new_node = node->right;
    }
    else
    {
        // otherwise make right child the rightmost leaf of the subtree rooted in the left child
        // and make the left child the new node
        Node * rightmost = new_node = node->left;
        while (rightmost->right != NULL)
            rightmost = rightmost->right;
        rightmost->right = node->right;
    }

    free(node);

    Node * new_root;
    if (parent == NULL)
    {
        // if deleted node was root, then return new node as root
        new_root = new_node;
    }
    else
    {
        // otherwise glue new node with parent and return old root
        new_root = root;
        if (data < parent->data)
            parent->left = new_node;
        else
            parent->right = new_node;
    }
    assert(isOrderedConcurrently(new_root));
    root = new_root;
    return;
}




void* ServeClient(char *client){

    // TODO: Open the file and read commands line by line

//    lockableNode * lockable = convertNodeToLockable();

    FILE * clientFile;
    clientFile = fopen(client, "r");

    char command[30];


    while (fscanf(clientFile, "%s", command) == 1)
    {
        // TODO: Handle command: "insertNode <some unsigned int value>"
        // print: "[ClientName]insertNode <SomeNumber>\n"
        // e.g. "[client1_commands]insertNode 1\n"
        if (strcmp(command, "insertNode") == 0)
        {
            fscanf(clientFile, "%s", command);

            pthread_rwlock_rdlock(&rwLock);
            insertNodeByRoot(atoi(command));
            printf("[%s]insertNode <%i>\n", client, atoi(command));
            pthread_rwlock_unlock(&rwLock);

//
        }
            // TODO: Handle command: "deleteNode <some unsigned int value>"
            // print: "[ClientName]deleteNode <SomeNumber>\n"
            // e.g. "[client1_commands]deleteNode 1\n"
        else if (strcmp(command, "deleteNode") == 0)
        {
            fscanf(clientFile, "%s", command);

            pthread_rwlock_wrlock(&rwLock);
            deleteNodeConcurrently(atoi(command));
            printf("[%s]deleteNode <%i>\n", client, atoi(command));
            pthread_rwlock_unlock(&rwLock);
        }
            // TODO: Handle command: "countNodes"
            // print: "[ClientName]countNodes = <SomeNumber>\n"
            // e.g. "[client1_commands]countNodes 1\n"
        else if(strcmp(command, "countNodes") == 0)
        {
            pthread_rwlock_rdlock(&rwLock);
            printf("[%s]countNodes = <%i>\n", client, countNodes(root));
            pthread_rwlock_unlock(&rwLock);
        }
            // TODO: Handle command: "sumSubtree"
            // print: "[ClientName]sumSubtree = <SomeNumber>\n"
            // e.g. "[client1_commands]sumSubtree 1\n"
        else if(strcmp(command, "sumSubTree"))
        {
            pthread_rwlock_rdlock(&rwLock);
            printf("[%s]sumSubtree = <%i>\n", client, sumSubtree(root));
            pthread_rwlock_unlock(&rwLock);
        }


    }

    fclose(clientFile);




    // TODO: match and execute commands











    return NULL;
    }

void *downtime(){

    sleep(1);
//    //TODO: 1st downtime: Do balanceTree here
    pthread_rwlock_wrlock(&rwLock);
    root = balanceTree(root);
    pthread_rwlock_unlock(&rwLock);
//
    sleep(1);
//    //TODO: 2nd downtime: Do balanceTree here
    pthread_rwlock_wrlock(&rwLock);
    root = balanceTree(root);
    pthread_rwlock_unlock(&rwLock);

    sleep(1);
//    //TODO: 3rd downtime: Do balanceTree here
    pthread_rwlock_wrlock(&rwLock);
    root = balanceTree(root);
    pthread_rwlock_unlock(&rwLock);

    return NULL;
}
