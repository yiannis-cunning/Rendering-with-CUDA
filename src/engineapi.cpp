// #include "engineapi.h"
#include "looper.h"
#include "renderer.h"
#include "common.h"
#include "engineapi.h"

static HANDLE msgloop_threadp, cudacore_threadp;
static DWORD msgloop_tid, cudacore_tid;


static command_t *cudacore_cmd = NULL;
static HANDLE cudacore_pending_sem, cudacore_done_sem;



void start_window(){
     thread_args_t *msgloop_args = NULL;
     thread_args_t *cudacore_args = NULL;
     void *dynamic_data;

     HANDLE sendcore_sem, rdy_to_send_sem;
     HANDLE mutex1;
     // 1) Common resources

     sendcore_sem = CreateSemaphore( NULL, 0, 1,NULL); // security stuff, initial count, maximum count, name stuff
     passert(sendcore_sem != NULL, "ENGINEAPI: Error init-ing semaphore");
     rdy_to_send_sem = CreateSemaphore( NULL, 0, 1,NULL); // security stuff, initial count, maximum count, name stuff
     passert(rdy_to_send_sem != NULL, "ENGINEAPI: Error init-ing semaphore");
     dynamic_data = calloc(sizeof(dynamic_render_data_t), 1);

     mutex1 = CreateMutexA(NULL, false, NULL);
     passert(mutex1 != NULL, "ENGINEAPI: Error init-ing mutex");


     // 2) thread for msg loop and 1 thread for launching cuda kernels

     msgloop_args = (thread_args_t *)calloc(sizeof(thread_args_t), 1);
     msgloop_args->sem1 = sendcore_sem;
     msgloop_args->sem2 = rdy_to_send_sem;
     msgloop_args->p1 = dynamic_data;
     msgloop_args->mutex1 = mutex1;
     
     msgloop_threadp = CreateThread( 
          NULL,                    // default security attributes
          0,                       // use default stack size  
          looper_init,             // thread function name
          msgloop_args,            // argument to thread function 
          0,                       // use default creation flags 
          &msgloop_tid);
     passert(msgloop_threadp != NULL, "ENGINEAPI: Error making msg loop thread");



     // 3) make thread for looper
     cudacore_args = (thread_args_t *)calloc(sizeof(thread_args_t), 1);
     cudacore_args->sem1 = sendcore_sem;
     cudacore_args->sem2 = rdy_to_send_sem;
     cudacore_args->p1 = dynamic_data;
     cudacore_args->mutex1 = mutex1;
     
     /* Adding command queue between API ---> renderer */
     cudacore_args->pending_cmd = CreateSemaphore( NULL, 0, 1, NULL);
     cudacore_cmd = (command_t *)calloc(sizeof(command_t), 1);
     cudacore_args->command = cudacore_cmd;
     cudacore_args->done_cmd = CreateSemaphore( NULL, 0, 1, NULL);


     cudacore_threadp = CreateThread( 
          NULL,                    // default security attributes
          0,                       // use default stack size  
          cuda_sender_init,        // thread function name
          cudacore_args,           // argument to thread function 
          0,                       // use default creation flags 
          &cudacore_tid);
     passert(cudacore_threadp != NULL, "ENGINEAPI: Error making cuda sender thread");
     

     printf("ENGINEAPI: Sent both threads... waiting for both to finish\n");

     //WaitForSingleObject(msgloop_threadp, INFINITE);
     //WaitForSingleObject(cudacore_threadp, INFINITE);
}


void quit_window(){
     WaitForSingleObject(msgloop_threadp, INFINITE);
     WaitForSingleObject(cudacore_threadp, INFINITE);
}



int add_asset(const char *filename){
     
     cudacore_cmd->type = CMD_ADD_ASSET;
     cudacore_cmd->filename = (char *)calloc(strlen(filename) + 1, 1);
     strcpy(cudacore_cmd->filename, filename);

     ReleaseSemaphore( cudacore_pending_sem, 1, NULL); // post a command
     WaitForSingleObject(cudacore_done_sem, INFINITE); // dec empty spots by 1
     return cudacore_cmd->ret_val;
}

/*
int main(int argc, char *argv[]){

     start_window();
     return 1;

}
*/