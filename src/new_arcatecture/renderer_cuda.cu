
#include "renderer_cuda.h"


void passert_cuda(bool cond, const char *msg){
       if(!cond){
              printf("CUDA ERROR: %s: %s\n", msg, cudaGetErrorString(cudaGetLastError()));
              exit(1);
       }
}


void safe_cudaAlloc(void **dest, uint32_t size, const char *msg){
       cudaMalloc(dest, size);
       if(*dest == NULL){
              printf("CUDA ALLOCATION ERROR: %s: %s\n", msg, cudaGetErrorString(cudaGetLastError()));
              exit(1);
       }
}




typedef struct asset_record_t {    // Linked list struct
       uint32_t             nTrigs;
       float                *d_trigs;
       uint8_t              *d_colors;
       asset_record_t       *next;
} asset_record_t;







typedef struct gpu_allocations_t{

       static_render_data_t *d_static_data;
       static_render_data_t static_data_local_copy;
       uint32_t             *d_depthScreen_arr;
       uint8_t              *d_pixels_arr;

       int                   asset_arr_size;
       int                   n_assets;
       asset_record_t       *asset_record_head;         // linked list of other allocations
       asset_t              *d_asset_pointer_arr;


       dynamic_render_data2_t      *d_dynamic_data;

       uint32_t                    n_instances;
       uint32_t                    sz_buffer;
       instance2_t                 *instance_pointer_arr;
       float                       *d_cord_arr_buffer;

} gpu_allocations_t;





static gpu_allocations_t *allocs;


void add_record(asset_record_t *head, asset_record_t *new_record){
       if(head == NULL){
              return ;
       }
       while(head->next != NULL){
              head = head->next;
       }
       head->next = new_record;
}

asset_record_t *get_nth_record(asset_record_t *head, int n){
       while(head != NULL && n != 0){
              n = n - 1;
              head = head->next;
       }
       return head;
}



int init(SDL_Surface *image){
       allocs = (gpu_allocations_t *)calloc(sizeof(gpu_allocations_t), 1);
       passert(allocs != NULL, "Initing cuda renderer");

       static_render_data_t srd;

       srd.pixels_arr       = NULL;
       srd.depthScreen_arr  = NULL;
       srd.BPP              = image->format->BytesPerPixel;
       srd.pitch            = image->pitch;
       srd.w                = image->w;
       srd.h                = image->h;
       srd.asset_pointer_arr= NULL;

       int pixel_arr_size = srd.pitch*srd.h*sizeof(Uint8);
       int depth_arr_size = srd.w*srd.h*sizeof(uint32_t);

       safe_cudaAlloc((void **)&(allocs->d_depthScreen_arr), depth_arr_size, "Depth array");

       safe_cudaAlloc((void **)&(allocs->d_pixels_arr), pixel_arr_size, "Pixel array");

       safe_cudaAlloc((void **)&(allocs->d_static_data), sizeof(static_render_data_t), "Static render data");

       safe_cudaAlloc((void **)&(allocs->d_dynamic_data), sizeof(dynamic_render_data2_t), "Dynamic render data");
       
       srd.depthScreen_arr = allocs->d_depthScreen_arr;
       srd.pixels_arr = allocs->d_pixels_arr;

       memcpy(&(allocs->static_data_local_copy), &srd, sizeof(static_render_data_t));
       cudaMemcpy(allocs->d_static_data, &srd, sizeof(static_render_data_t), cudaMemcpyHostToDevice);

       return 0;
}

int alloc_asset(float *trigs, uint8_t *colors, uint32_t nTrigs){

       // 1) Allocate space for new asset (a) space in array for pointer - maybe, (b) trig and color arr - always
       if(allocs->asset_arr_size == allocs->n_assets){
              // Need to resize array
              asset_t *new_allocation;
              int old_arr_size = allocs->n_assets;
              int new_arr_size = allocs->n_assets * 2 + 1;

              safe_cudaAlloc((void **)&(new_allocation), sizeof(asset_t)*new_arr_size, "Resizing asset array");
              cudaMemcpy(new_allocation, allocs->d_asset_pointer_arr, old_arr_size, cudaMemcpyDeviceToDevice); // Making own kernel would be faster apperently 
              if(allocs->d_asset_pointer_arr != NULL){
                     cudaFree(allocs->d_asset_pointer_arr);
              }
              allocs->d_asset_pointer_arr = new_allocation;
              
              // Copy over the new sigular pointer to the asset array
              allocs->static_data_local_copy.asset_pointer_arr = new_allocation;
              cudaMemcpy(allocs->d_static_data, &(allocs->static_data_local_copy), sizeof(static_render_data_t), cudaMemcpyHostToDevice);

       }
       allocs->n_assets = allocs->n_assets + 1;


       // Either way will need to allocate new gpu space for trigs and colors
       int trig_arr_size = nTrigs * sizeof(float) * 9;
       int color_arr_size = nTrigs * sizeof(uint8_t) * 3;

       // keep the record of the allocation
       asset_record_t *new_asset_record = (asset_record_t *)calloc(sizeof(asset_record_t), 1);
       new_asset_record->nTrigs = nTrigs;
       new_asset_record->next = NULL;
       if(allocs->asset_record_head == NULL){
              allocs->asset_record_head = new_asset_record;
       } else{
              add_record(allocs->asset_record_head, new_asset_record);
       
       }
       safe_cudaAlloc((void **)&(new_asset_record->d_trigs), trig_arr_size, "New asset trianlge array");
       safe_cudaAlloc((void **)&(new_asset_record->d_colors), color_arr_size, "New asset color array");


       asset_t gpu_asset;
       gpu_asset.nTrigs = nTrigs;
       gpu_asset.trigs = new_asset_record->d_trigs;
       gpu_asset.colors = new_asset_record->d_colors;


       // 2) copy over the data 
       cudaMemcpy(gpu_asset.trigs, trigs, trig_arr_size, cudaMemcpyHostToDevice);
       cudaMemcpy(gpu_asset.colors, colors, color_arr_size, cudaMemcpyHostToDevice);
       cudaMemcpy(allocs->d_asset_pointer_arr + allocs->n_assets - 1, &gpu_asset, sizeof(asset_t), cudaMemcpyHostToDevice);
       

       return 0;
}


// Takes in positon data + list of instances
int update_dynamic_data(float *view, float *offset, instance_t *inst_data, int n_instances){
       //cudaMemcpy(allocs->d_dynamic_data, dyn_rend_data, sizeof(dynamic_render_data2_t), cudaMemcpyHostToDevice);
       

       // 1) copy over position data to buffer
       dynamic_render_data2_t new_dynamic_rd = {0};
       cpyVec(view, new_dynamic_rd.view);
       cpyVec(offset, new_dynamic_rd.offset);

       new_dynamic_rd.instances_arr = allocs->instance_pointer_arr;
       
       instance_t *head = inst_data;
       if(head == NULL){
              cudaMemcpy(allocs->d_dynamic_data, &new_dynamic_rd, sizeof(dynamic_render_data2_t), cudaMemcpyHostToDevice);
              return 0;
       }

       // 2) Get the size of buffer needed for all these instances -> pointer array size is n_instances*sizeof(instance2_t)
       asset_record_t *asst_head = allocs->asset_record_head;
       asset_record_t *asset;
       uint32_t inst_sum;
       int i = 0;
       while(head != NULL && i < n_instances){
              asset = get_nth_record(asst_head, head->asset_id);
              if(asset == NULL){
                     return -1;    // Refering to a unkown asset
              }
              inst_sum = asset->nTrigs*9;
              head = head->next;
              i += 1;
       }
       if(n_instances != i){
              return -1;
       }

       // 3) Make a bigger buffer if needed - cord_arr
       if(allocs->sz_buffer < inst_sum){
              float *new_allocation_cord = NULL;
              safe_cudaAlloc((void **)&(new_allocation_cord), inst_sum*sizeof(float), "Resizing cord/buffer array");
              if(allocs->d_cord_arr_buffer != NULL){
                     cudaFree(allocs->d_cord_arr_buffer);
              }
              allocs->sz_buffer = inst_sum*sizeof(float);
              allocs->d_cord_arr_buffer = new_allocation_cord;
       }

       // 4) Make a bigger buffer if needed - pointer arr
       if(allocs->n_instances < n_instances){
              instance2_t *new_allocation_inst = NULL;
              safe_cudaAlloc((void **)&(new_allocation_inst), n_instances*sizeof(instance2_t), "Resizing instnace array");
              
              if(allocs->instance_pointer_arr != NULL){
                     cudaFree(allocs->instance_pointer_arr);
              }
              allocs->n_instances = n_instances;
              allocs->instance_pointer_arr = new_allocation_inst;
       }

       // 5) Now allocate buffer space for each instance - offsets into big block of memory - and also copy over the pointers
       new_dynamic_rd.instances_arr = allocs->instance_pointer_arr;
       
       instance2_t new_inst;
       uint32_t running_offset = 0;
       head = inst_data;
       for(int i = 0; i < n_instances; i += 1){
              new_inst.asset_id = head->asset_id;
              cpyVec(head->offset, new_inst.offset);
              new_inst.buffer_loc = allocs->d_cord_arr_buffer + running_offset;

              asset = get_nth_record(asst_head, head->asset_id);
              passert(asset != NULL, "Error with code");

              running_offset = running_offset + asset->nTrigs*9;
              head = head->next;
              cudaMemcpy(allocs->instance_pointer_arr + i, &new_inst, sizeof(instance2_t), cudaMemcpyHostToDevice);

       }

       cudaMemcpy(allocs->d_dynamic_data, &new_dynamic_rd, sizeof(dynamic_render_data2_t), cudaMemcpyHostToDevice);

       return 0;
}



void *get_device_trig_pointer(){
       if(allocs == NULL){
              return NULL;
       }
       passert(allocs->asset_record_head != NULL, "There are no asset records");
       return (void *)allocs->asset_record_head->d_trigs;
}


void *get_static_render_data_p(){
       return (void *)allocs->d_static_data;
}

void *get_dynamic_render_data_p(){
       return (void *)allocs->d_dynamic_data;
}
/*
[static data]
       [depth], [pixel], [assetp, assetp, assetp]

[dynamic data]



*/

int kill(){
       cudaFree(allocs->d_static_data);
       cudaFree(allocs->d_depthScreen_arr);
       cudaFree(allocs->d_pixels_arr);
       cudaFree(allocs->d_asset_pointer_arr);
       cudaFree(allocs->d_dynamic_data);


       asset_record_t *cur_asset_rec = allocs->asset_record_head;
       asset_record_t *prev_asset_rec;

       while(cur_asset_rec != NULL){
              cudaFree(cur_asset_rec->d_trigs);    // Free the trig array
              cudaFree(cur_asset_rec->d_colors);   // Free color array
              prev_asset_rec = cur_asset_rec;
              cur_asset_rec = cur_asset_rec->next;
              free(prev_asset_rec);              // Free this record
       }
       free(allocs);
       return 0;
}