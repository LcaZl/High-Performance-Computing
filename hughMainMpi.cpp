#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <ctype.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <err.h>
#include <math.h>
#include <mpi.h>
#include <assert.h>

// acumulator result ratio
#define RRATIO 2
#define TRATIO 2

/* start of utility functions: not interesting */
typedef unsigned char uchar;
typedef unsigned long ulong;

// struct for the image handling
typedef struct intensity_t {
	double **pix;
	long width, height;
} *intensity;

double PI;

#define decl_array_alloc_d(double) \
double ** double##_array(long w, long h) {			\
	int i;						\
	double ** row = (double **) malloc(sizeof(double*) * h);	\
	double *  pix = (double *) malloc(sizeof(double) * h * w);	\
	for (i = 0; i < h; i++)				\
		row[i] = pix + w * i;			\
	memset(pix, 0, sizeof(double) * h * w);		\
	return row;					\
}

#define decl_array_alloc_u(ulong) \
ulong ** ulong##_array(long w, long h) {			\
	int i;						\
	ulong ** row = (ulong **) malloc(sizeof(ulong*) * h);	\
	ulong *  pix = (ulong *) malloc(sizeof(ulong) * h * w);	\
	for (i = 0; i < h; i++)				\
		row[i] = pix + w * i;			\
	memset(pix, 0, sizeof(ulong) * h * w);		\
	return row;					\
}

decl_array_alloc_d(double);
decl_array_alloc_u(ulong);

intensity_t* intensity_alloc(long w, long h)
{
	intensity_t* x = (intensity_t *) malloc(sizeof(struct intensity_t));
	x->width = w;
	x->height = h;
	x->pix = double_array(w, h);

	return x;
}

long get_num(uchar **p, uchar *buf_end)
{
	uchar *ptr = *p, *tok_end;
	long tok;
	while (1) {
		while (ptr < buf_end && isspace(*ptr)) ptr++;
		if (ptr >= buf_end) return 0;

		if (*ptr == '#') { /* ignore comment */
			while (ptr++ < buf_end) {
				if (*ptr == '\n' || *ptr == '\r') break;
			}
			continue;
		}

		tok = strtol((char*)ptr, (char**)&tok_end, 10);
		if (tok_end == ptr) return 0;
		*p = tok_end;
		return tok;
	}
	return 0;
}

void write_pgm(double **pix, long w, long h, const char * name)
{
	long i, j;
	unsigned char *ptr, *buf = (uchar *) malloc(sizeof(double) * w * h);
	char header[1024];
	sprintf(header, "P5\n%ld %ld\n255\n", w, h);

	ptr = buf;
	for (i = 0; i < h; i++)
		for (j = 0; j < w; j++)
			*(ptr++) = 255 * pix[i][j];

	FILE *fptr;

   if ((fptr = fopen(name, "a+")) == NULL){
       printf("Error! opening file");

       // Program exits if the file pointer returns NULL.
       exit(1);
   }
   fclose(fptr);
	freopen(name, "wb", stdout);
	write(fileno(stdout), header, strlen(header));
	write(fileno(stdout), buf, w * h);

	free(buf);

}

intensity read_pnm(char *name)
{
	struct stat st;
	uchar *fbuf, *ptr, *end;
	long width, height, max_val;
	int i, j;
	intensity ret;

	int fd = open(name, O_RDONLY);
	if (fd == -1) err(1, "Can't open %s", name);

    // error handling in reading
	fstat(fd, &st);
	fbuf = (uchar *) malloc(st.st_size + 1);
	read(fd, fbuf, st.st_size);
	*(end = fbuf + st.st_size) = '\0';
	close(fd);

	if (fbuf[0] != 'P' || (fbuf[1] != '5' && fbuf[1] != '6') || !isspace(fbuf[2]))
		err(1, "%s: bad format: can only do P5 or P6 pnm", name);

	ptr = fbuf + 3;
	width   = get_num(&ptr, end);
	height  = get_num(&ptr, end);
	max_val = get_num(&ptr, end);
	if (max_val <= 0 || max_val >= 256)
		err(1, "Can't handle pixel value %ld\n", max_val);

	// ret is assigned
	ret = intensity_alloc(width, height);
	ptr ++;

	// graymaping
	double acum = 0;
	if (fbuf[1] == '5') {	
		// 1 byte per pixel
		for (i = 0; i < height; i++)
		{
			for (j = 0; j < width; j++)
			{
				double pixVal = (double)*(ptr++) / max_val;
				ret->pix[i][j] = pixVal;
				acum += pixVal;
				printf("%lf ", pixVal);
			}
			printf("\n black\n");

		}
	} 
	else 
	{		
		// 1 byte each for RGB 
		for (i = 0; i < height; i++)
		{
			for (j = 0; j < width; j++)
			{
				double pixVal = (ptr[0] * 0.2126 + ptr[1] * 0.7152 + ptr[2] * 0.0722) / 255;
				ret->pix[i][j] = pixVal;
				acum += pixVal;
				ptr += 3;
			}
		}
	}

	// threshold tranf
	ptr = fbuf + 3;
	ptr ++;	
	float threshold = acum / (ret->width * ret->height);
	
	intensity out;
	out = intensity_alloc(width, height);

	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			if (ret->pix[i][j] > threshold)
			{
				ret->pix[i][j] = 1;
			}
			else
			{
				ret->pix[i][j] = 0;
			}
			out->pix[i][j] = 0;
		}
	}
	int mHeight = height - 1;
	int mWidth = width - 1;
	// start step that leaves only borders
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			if(j == 0 || j == mWidth)
			{
				if(j == 0)
				{
					if(i == 0 || i == mHeight)
					{
						if(i == 0)
						{
							if(ret->pix[i][j] == 1)
							{
								if (ret->pix[i + 1][j] == 1)
								{
									if (ret->pix[i][j + 1] == 1)
									{
										out->pix[i][j] = 1;
									}
									else out->pix[i][j] = 0;
								}
								else out->pix[i][j] = 0;
							}
							else out->pix[i][j] = 0;
						}
						if(i == mHeight)
						{
							if(ret->pix[i][j] == 1)
							{
								if (ret->pix[i - 1][j] == 1)
								{
									if (ret->pix[i][j + 1] == 1)
									{
										out->pix[i][j] = 1;
									}
									else out->pix[i][j] = 0;
								}
								else out->pix[i][j] = 0;
							}
							else out->pix[i][j] = 0;
						}
					}
				}
				if(j == mWidth)
				{
					if(i == 0 || i == mHeight)
					{
						if(i == 0)
						{
							if(ret->pix[i][j] == 1)
							{
								if (ret->pix[i + 1][j] == 1)
								{
									if (ret->pix[i][j - 1] == 1)
									{
										out->pix[i][j] = 1;
									}
									else out->pix[i][j] = 0;
								}
								else out->pix[i][j] = 0;
							}
							else out->pix[i][j] = 0;
						}
						if(i == mHeight)
						{
							if(ret->pix[i][j] == 1)
							{
								if (ret->pix[i - 1][j] == 1)
								{
									if (ret->pix[i][j - 1] == 1)
									{
										out->pix[i][j] = 1;
									}
									else out->pix[i][j] = 0;
								}
								else out->pix[i][j] = 0;
							}
							else out->pix[i][j] = 0;
						}
					}
				}
			}
			else if(i == 0 || i == mHeight)
			{
				if(i == 0)
				{
					if(ret->pix[i][j] == 1)
					{
						if (ret->pix[i + 1][j] == 1)
						{
							if (ret->pix[i][j + 1] == 1)
							{
								if (ret->pix[i][j - 1] == 1)
								{
									out->pix[i][j] = 1;
								}
								else out->pix[i][j] = 0;
							}
							else out->pix[i][j] = 0;				
						}
						else out->pix[i][j] = 0;
					}
					else out->pix[i][j] = 0;
				}
				if(i == mHeight)
				{
					if(ret->pix[i][j] == 1)
					{
						if (ret->pix[i - 1][j] == 1)
						{
							if (ret->pix[i][j + 1] == 1)
							{
								if (ret->pix[i][j - 1] == 1)
								{
									out->pix[i][j] = 1;
								}
								else out->pix[i][j] = 0;
							}
							else out->pix[i][j] = 0;				
						}
						else out->pix[i][j] = 0;
					}
					else out->pix[i][j] = 0;
				}
			}
			else
			{
				if(ret->pix[i][j] == 1)
				{
					if (ret->pix[i + 1][j] == 1)
					{
						if (ret->pix[i - 1][j] == 1)
						{
							if (ret->pix[i][j + 1] == 1)
							{
								if (ret->pix[i][j - 1] == 1)
								{
									out->pix[i][j] = 1;
								}
								else out->pix[i][j] = 0;
							}
							else out->pix[i][j] = 0;				
						}
						else out->pix[i][j] = 0;
					}
					else out->pix[i][j] = 0;
				}
				else out->pix[i][j] = 0;
			}

		}

    }


	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			if(ret->pix[i][j] == 1 && out->pix[i][j] == 0)
			{
				ret->pix[i][j] = 1;
			}
			else
			{
				ret->pix[i][j] = 0;
			}
		}
	}
	// finishes step that leaves only borders

	free(fbuf);
	return ret;
}

/* Finally, end of util functions.  All that for this function. */
intensity hugh_transform(intensity in, double gamma)
{
	long i, j, w, h;
	double r_res, t_res, r, x, y, max_val, min_val, *pp;
	double t1, t2, tmin;

	// calculate PI
	PI = atan2(1, 1) * 4;

	x = in->width - .5;
	y = in->height - .5;
	r = sqrt(x * x + y * y) / 2;

	w = in->width / TRATIO;
	h = in->height / RRATIO;
	r_res = r / h;
	t_res = PI * 2 / w;
	intensity global_rez;
	global_rez = intensity_alloc(w, h);
	
	t1 = MPI_Wtime();

	MPI_Init(NULL, NULL);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	
	
	double theta, rho, my_y, my_x, my_r, k; 
	long my_m, my_l;
	int leng_of_proc;
	leng_of_proc = in->height / world_size;
	intensity graph;
	graph = intensity_alloc(w, h);

	fprintf(stderr, "%d, %d, %d, %d, %d\n", world_size, world_rank, leng_of_proc, leng_of_proc * world_rank, (leng_of_proc * world_rank) + leng_of_proc);

	// hough loop
	for (i = leng_of_proc * world_rank; i < (leng_of_proc * world_rank) + leng_of_proc; i++) {
		my_y = i - in->height / 2. + .5;
		for (j = 0; j < in->width; j++) {
			my_x = j - in->width / 2 + .5;
			my_r = sqrt(my_x * my_x + my_y * my_y);

			// at each pixel, check what lines it could be on
			for (k = 0; k < w; k++) {
				theta = k * t_res - PI;
				rho = my_x * cos(theta) + my_y * sin(theta);
				if (rho >= 0) {
					my_m = rho / r_res;
					my_l = k;
				} else {
					my_m = -rho / r_res;
					my_l = (k + w/2.);
					my_l %= w;
				}
				graph->pix[my_m][my_l] += in->pix[i][j] * r;
			}
		}
		// show which row we are precessing lest user gets bored
		if(world_rank == 0)
		{
			fprintf(stderr, "\r%ld", i);
		}
	}

	/* ********************* TODO ********************* */
	// join the acumulator of each process

	// for (i = leng_of_proc * world_rank; i < (leng_of_proc * world_rank) + leng_of_proc; i++)
	// {
	// 	MPI_Reduce(&(graph->pix[i]), &(global_rez->pix[i]), graph->height * graph->width, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	// }

	if(world_rank == 0)
	{
		fprintf(stderr, "Finished\n");
	}
	MPI_Finalize();
	max_val = 0;
	min_val = 1e100;
	pp = &(global_rez->pix[global_rez->height - 1][global_rez->width - 1]);

	for (i = global_rez->height * global_rez->width - 1; i >= 0; i--, pp--)
	{
		if (max_val < *pp) max_val = *pp;
		if (min_val > *pp) min_val = *pp;
	}

	/* gamma correction. if gamma > 1, output contrast is better, noise
	   is suppressed, but spots for thin lines may be lost; if gamma < 1,
	   everything is brighter, both lines and noises */
	pp = &(global_rez->pix[global_rez->height - 1][global_rez->width - 1]);
	for (i = global_rez->height * global_rez->width - 1; i >= 0; i--, pp--) {
		*pp = pow((*pp - min_val)/ (max_val - min_val), gamma);
	}

	return global_rez;
}

int main(int argc, char* argv[])
{

	if (argc < 3)
	{
		printf("Not enought arguments.\nYou should add the path of the input image and then add the output.");
		return 1;
	}
	
	// reads pnm files and make them bw
	intensity in = read_pnm(argv[1]);

	write_pgm(in->pix, in->width, in->height, "black&white.pnm");

	intensity out = hugh_transform(in, 1.5);

	write_pgm(out->pix, out->width, out->height, argv[2]);

        /* not going to free memory we used: OS can deal with it */
	return 0;
}