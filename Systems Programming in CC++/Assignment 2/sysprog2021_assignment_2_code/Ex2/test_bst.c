
#include<pthread.h>
#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include <unistd.h>

#include "bst.h"

Node *root=NULL;
Node *root_balanced=NULL;

#include "unique_rng.c"
#include "serve_client.c"

#ifdef CYCLE_TEST
#include "cpucycles.c"
#endif

int f_verbose = 0;

void clean(){
	/*****************************************************/
	/******   Free resources before program ends  ********/
	/*****************************************************/

	root=freeSubtree(root);
	root_balanced = freeSubtree(root_balanced);
	root= NULL;
	root_balanced = NULL;
	return;
}

#ifdef CYCLE_TEST
// DO NOT add any test functions here, as they will not be compiled
void test_cycle(){
	unsigned int COUNTER, REPEAT=1000;
	unsigned int i, r, sum, sum_balanced;

	init_rand();

	// Create a random BST rooted at 'root'
	for(i=0; i<10000; i++){
		r = unique_random_number();
		root=insertNode(root, r);
	}

	long int CLOCK_total, CLOCK_end, CLOCK_start;
	// Create a balanced BST from the unbalanced BST
	root_balanced = balanceTree(root);


	// Cycle count for sum on unbalanced BST
	CLOCK_total=0;
	for(COUNTER=0; COUNTER<REPEAT; COUNTER++)
	{
		CLOCK_start = cpucycles();
		sum = sumSubtree(root);
		CLOCK_end = cpucycles();
		CLOCK_total = CLOCK_total + CLOCK_end - CLOCK_start;
	}
	printf("Avg cycle count for unbalanced BST = %ld\n", CLOCK_total/REPEAT);
	printf("Sum of unbalanced BST = %u\n", sum);

	// Cycle count for sum on balanced BST
	CLOCK_total=0;
	for(COUNTER=0; COUNTER<REPEAT; COUNTER++)
	{
		CLOCK_start = cpucycles();
		sum_balanced = sumSubtree(root_balanced);
		CLOCK_end = cpucycles();
		CLOCK_total = CLOCK_total + CLOCK_end - CLOCK_start;
	}
	printf("Avg cycle count for balanced BST = %ld\n", CLOCK_total/REPEAT);
	printf("Sum of balanced BST = %u\n", sum_balanced);

	printf("Difference in sums = %d\n", sum - sum_balanced);

	clean();

}
#else
void test_task12(){
	unsigned int i,r, sum, sum_balanced;
	int failed = 0;
	init_rand();
	// Create a random BST rooted at 'root'
	for(i=0; i<10000; i++){
		r = unique_random_number();     // This will give you the same set of random numbers every time
		root=insertNode(root, r);
	}

	/*****************************************************/
	/******   Part 1 of Exercise 2 Starts here    ********/
	/*****************************************************/

	printf("/******** TEST OF PART 1 ********/\n\n");

	// Create a balanced BST from the unbalanced BST
	root_balanced = balanceTree(root);

	// Sum on unbalanced BST

	sum = sumSubtree(root);

	printf("Sum of unbalanced BST = %u\n", sum);

	// Sum on balanced BST
	sum_balanced = sumSubtree(root_balanced);

	printf("Sum of balanced BST = %u\n", sum_balanced);

	printf("Difference in sums = %d\n", sum - sum_balanced);
	destroy_rand();

	failed = (sum - sum_balanced) + (sum - 289060379);
	
	if(failed & f_verbose)
	{
	  printf("\n/******** START DEBUG MODE *********/\n");

	  printf("\nUnbalanced tree:\n");
	  printf("label1\n");
	  printSubtree(root);
	  printf("label2\n");
	  printf("\nBalanced tree:\n");
	  printf("label3\n");
	  printSubtree(root_balanced);
	  printf("label4\n");
	  printf("\n/******** END DEBUG MODE ********/\n");
	}

	printf("\n /******** END OF PART 1 ********/\n\n");

	clean();
}

void test_tack34(){

	printf("/******** TEST OF PART 2 ********/\n\n");
	unsigned int i;
	char *client_names[5] = {"client1_commands", "client2_commands", "client3_commands",
	                         "client4_commands", "client5_commands"};

	pthread_t threads[6];

	/*****************************************************/
	/******   Part 2 of Exercise 2 Starts here    ********/
	/*****************************************************/

	// spawn all threads
	pthread_create(&threads[0], NULL, (void *) ServeClient, client_names[0]);
	pthread_create(&threads[1], NULL, (void *) ServeClient, client_names[1]);
	pthread_create(&threads[2], NULL, (void *) ServeClient, client_names[2]);
	pthread_create(&threads[3], NULL, (void *) ServeClient, client_names[3]);
	pthread_create(&threads[4], NULL, (void *) ServeClient, client_names[4]);
	pthread_create(&threads[5], NULL, (void *) downtime, NULL);


	// join all readers
	for (i = 0; i < 6; i++) {
		pthread_join(threads[i], NULL);
	}

	// The tree should only have one node now
	int count = countNodes(root);
	int sum = sumSubtree(root);

	if (count == 1 && sum == 1){
		printf("Test for Part2 seems OK\n");
	} else{
		printf("Test for Part2 fail\n");
	}

	// Free the tree
	clean();
}

// TODO: You could add more test functions here

#endif
int main(int argc, char *argv[]){

	if(argc == 2){
	  if(strcmp(argv[1],"-v") == 0)
	    {
	      f_verbose = 1;
	    }
 	}

#ifdef CYCLE_TEST
	// DO NOT add any test functions here, as they will not be compiled
	test_cycle();
#else
	test_task12();
	test_tack34();

	// TODO: You could call your test functions at here

#endif
	return 0;
}
