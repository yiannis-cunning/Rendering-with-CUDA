#include "common.h"






void passert(bool tf, const char *msg){
       if(tf){
              return;
       }
       printf("%s\n", msg);
       exit(1);
}