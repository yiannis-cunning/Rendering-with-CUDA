#include "readSTL.h"
#include <stdint.h>

bool load_triangles_stl(char *filename, float **trigs, char **clrs, int *nTrigs){
       FILE *fptr;
       if ((fptr = fopen(filename,"rb")) == NULL){printf("Error! opening stl file (%s)", filename); return(false);}
       
       // Seek past the header (80 BYTES) and get the number of trigs
       char *data = (char *)malloc(sizeof(float) * 9);
       fseek(fptr, 80, SEEK_CUR);
       fread(data, 1, 4, fptr);
       int nTrigs_local = *((int *)data);

       // Allocate space for objects
       float *trigs_local = (float *)malloc(sizeof(float)*3*3*nTrigs_local);
       char *clrs_local = (char *)malloc(nTrigs_local*3);
       if(trigs_local == NULL || clrs_local == NULL){printf("Error allocating space");return false;}

       // Loop for each triangle in the file to fill in data
       for(int i = 0; i < nTrigs_local; i ++){
              fseek(fptr, 12, SEEK_CUR);
              fread(trigs_local + i*9, 4, 9, fptr);
              clrs_local[i*3] = 0xE2;
              clrs_local[i*3 + 1] = 0xA3;
              clrs_local[i*3 + 2] = 0x1F;
              fseek(fptr, 2, SEEK_CUR);
       }
       
       *trigs = trigs_local;
       *clrs = clrs_local;
       *nTrigs = nTrigs_local;

       fclose(fptr);
       free(data);
       return true;
}


bool read_bitmap(char *filename, char *bits){
       FILE *fptr;
       if ((fptr = fopen(filename,"rb")) == NULL){printf("Error! opening bitmap file (%s)", filename); return(false);}
       
       // File Type
       char d[2];
       int n;
       short x;


       fread(d, 1, 2, fptr);
       printf("File Type: %c-%c\n", d[0], d[1]);        // bits 0 - 1 are file type
       fread(&n, sizeof(int), 1, fptr);
       printf("Size of BMP: %d\n", n);                  // bits 2 - 5 are size
       fread(&n, 4, 1, fptr);
       printf("Nothing: %d\n", n);                      // bits 6 - 9 are nothing
       fread(&n, 4, 1, fptr);
       printf("BMP offset: %d\n", n);                   // Bits 10-13 are offset to pixels
       fread(&n, 4, 1, fptr);
       printf("DIB Header size: %d\n", n);              // bits 14-19 are DIBH size
       
       fread(&n, 4, 1, fptr);printf("Width: %d\n", n);
       fread(&n, 4, 1, fptr);printf("Height: %d\n", n);
       fread(&x, 2, 1, fptr);printf("Color planes: %d\n", x);
       fread(&x, 2, 1, fptr);printf("Bits Per pixel: %d\n", x);
       fread(&n, 4, 1, fptr);fread(&n, 4, 1, fptr);fread(&n, 4, 1, fptr);fread(&n, 4, 1, fptr);fread(&n, 4, 1, fptr);fread(&n, 4, 1, fptr);
       unsigned char c;
       for(int i = 0; i < 3; i++){c = fgetc(fptr); printf("%X ", c);} printf("\n");
       for(int i = 0; i < 3; i++){c = fgetc(fptr); printf("%X ", c);} printf("\n");
       for(int i = 0; i < 2; i++){c = fgetc(fptr); printf("%X ", c);} printf("\n");
       for(int i = 0; i < 3; i++){c = fgetc(fptr); printf("%X ", c);} printf("\n");
       for(int i = 0; i < 3; i++){c = fgetc(fptr); printf("%X ", c);} printf("\n");
       for(int i = 0; i < 2; i++){c = fgetc(fptr); printf("%X ", c);} printf("\n");

       /*
       uint8_t c;
       for(int i = 0; i < 16; i++){
              c = fgetc(fptr);
              printf("%X ", c);
              if(i == 3 || i == 11){printf("\t");}
              if(i == 7){printf("\n");}
       }
       c = fgetc(fptr);*/
       /*
       char c = fgetc(fptr);
       while(c != EOF){
              printf("%x \n", c);
              c = fgetc(fptr);
       }*/
       fclose(fptr);
       return true;
}

int main(){
       char filename[20] = {"thing.bmp"};
       char *bits;
       read_bitmap(filename, bits);
       return 0;

}