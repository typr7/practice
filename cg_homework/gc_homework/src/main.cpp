#include <omp.h>
#include <stdio.h>


 int HelloFunc()
{
    int i;
    int numthreads = 8;
#pragma omp parallel for default(none) num_threads(numthreads) private(i)
    for (i = 0; i < 100; i++)
    {
        int tid = omp_get_thread_num();
        printf("Hello world from omp thread %d\n", tid);
    }
    return -1;
}

int main()
{
    HelloFunc();

  return 0;
}
