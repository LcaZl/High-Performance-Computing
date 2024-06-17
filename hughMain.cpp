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

// more or less base code
/* start of utility functions: not interesting */
typedef unsigned char uchar;
typedef unsigned long ulong;
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

/* Note: not robust. A robust version would be to long for example code */
intensity read_pnm(char *name)
{

	struct stat st;
	uchar *fbuf, *ptr, *end;
	long width, height, max_val;
	int i, j;
	intensity ret;

	int fd = open(name, O_RDONLY);
	if (fd == -1) err(1, "Can't open %s", name);

	/* from now on assume all operations succeed */
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

	fprintf(stderr, "[Info] format: P%c w: %ld h: %ld value: %ld\n",
		fbuf[1], width, height, max_val);

	ret = intensity_alloc(width, height);
	ptr ++;	/* ptr should be pointint at the first pixel byte now */

	if (fbuf[1] == '5') {	/* graymap, 1 byte per pixel */
		for (i = 0; i < height; i++) {
			for (j = 0; j < width; j++) {
				ret->pix[i][j] = (double)*(ptr++) / max_val;
			}
		}
	} else {		
		// pnm, 1 byte each for RGB
		/* hocus pocus way of getting lightness from RGB for us */
		for (i = 0; i < height; i++) {
			for (j = 0; j < width; j++) {
				ret->pix[i][j] = (ptr[0] * 0.2126 +
						  ptr[1] * 0.7152 +
						  ptr[2] * 0.0722) / max_val;
				ptr += 3;
			}
		}
	}

	free(fbuf);
	return ret;
}

void write_pgm(double **pix, long w, long h)
{
	long i, j;
	unsigned char *ptr, *buf = (uchar *) malloc(sizeof(double) * w * h);
	char header[1024];
	sprintf(header, "P5\n%ld %ld\n255\n", w, h);

	ptr = buf;
	for (i = 0; i < h; i++)
		for (j = 0; j < w; j++)
			*(ptr++) = 256 * pix[i][j];

	FILE *fptr;

   if ((fptr = fopen("result.pnm","a+")) == NULL){
       printf("Error! opening file");

       // Program exits if the file pointer returns NULL.
       exit(1);
   }
   fclose(fptr);
	freopen("result.pnm", "wb", stdout);
	write(fileno(stdout), header, strlen(header));
	write(fileno(stdout), buf, w * h);

	free(buf);

}

/* Finally, end of util functions.  All that for this function. */
intensity hugh_transform(intensity in, double gamma)
{
	long i, j, k, l, m, w, h;
	double bg, r_res, t_res, rho, r, theta, x, y, v, max_val, min_val, *pp;
	intensity graph;

	/* before anything else, legalize Pi = 3 */
	PI = atan2(1, 1) * 4;

	/* first, run through all pixels and see what the average is,
	 * so we can take a guess if the background is black or white.
	 * a real application wouldn't do silly things like this	*/
	for (i = 0, bg = 0; i < in->height; i++)
		for (j = 0; j < in->width; j++)
			bg += in->pix[i][j];
	fprintf(stderr, "[info] background is %f\n", bg);
	bg = (bg /= (in->height * in->width) > 0.5) ? 1 : 0;

	/* if white, invert it */
	if (bg) {
		for (i = 0; i < in->height; i++)
			for (j = 0; j < in->width; j++)
				in->pix[i][j] = 1 - in->pix[i][j];
	}

	/* second, decide what resolution of rho and theta should be.
	 * here we just make the rho/theta graph a fixed ratio
	 * of input, which is dumb.  It should depend on the application.
	 * finer bins allow better resolution between lines, but will
	 * lose contrast if the input is noisy.  Also, lower resolution, faster.
	 */
#	define RRATIO 1.5
#	define TRATIO 1.5
	x = in->width - .5;
	y = in->height - .5;
	r = sqrt(x * x + y * y) / 2;

	w = in->width / TRATIO;
	h = in->height / RRATIO;
	r_res = r / h;
	t_res = PI * 2 / w;

	graph = intensity_alloc(w, h);

	for (i = 0; i < in->height; i++) {
		y = i - in->height / 2. + .5;
		for (j = 0; j < in->width; j++) {
			x = j - in->width / 2 + .5;
			r = sqrt(x * x + y * y);
			v = in->pix[i][j];

			/* hackery: sample image is mostly blank, this saves a great
			 * deal of time.  Doesn't help a lot with noisy images */
			if (!v) continue;

			/* at each pixel, check what lines it could be on */
			for (k = 0; k < w; k++) {
				theta = k * t_res - PI;
				rho = x * cos(theta) + y * sin(theta);
				if (rho >= 0) {
					m = rho / r_res;
					l = k;
				} else {
					m = -rho / r_res;
					l = (k + w/2.);
					l %= w;
				}
				graph->pix[m][l] += v * r;
			}
		}
		/* show which row we are precessing lest user gets bored */
		fprintf(stderr, "\r%ld", i);
	}
	fprintf(stderr, "\n");

	max_val = 0;
	min_val = 1e100;
	pp = &(graph->pix[graph->height - 1][graph->width - 1]);
	for (i = graph->height * graph->width - 1; i >= 0; i--, pp--) {
		if (max_val < *pp) max_val = *pp;
		if (min_val > *pp) min_val = *pp;
	}

	/* gamma correction. if gamma > 1, output contrast is better, noise
	   is suppressed, but spots for thin lines may be lost; if gamma < 1,
	   everything is brighter, both lines and noises */
	pp = &(graph->pix[graph->height - 1][graph->width - 1]);
	for (i = graph->height * graph->width - 1; i >= 0; i--, pp--) {
		*pp = pow((*pp - min_val)/ (max_val - min_val), gamma);
	}

	return graph;
}

int main()
{
	// reads pnm files
	// used pnm files as it was the available in the code I used as a base
	intensity in = read_pnm("images/1-1.pnm");


	intensity out = hugh_transform(in, 1.5);

	/* binary output goes straight to stdout, get ready to see garbage on your
	 * screen if you are not careful!
	 */
	write_pgm(out->pix, out->width, out->height);

        /* not going to free memory we used: OS can deal with it */
	return 0;
}