#include "kmeans.h"
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>
#include <CL/cl.h>

#define CHECK_ERROR(err) \
  if (err != CL_SUCCESS) { \
    printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
    exit(EXIT_FAILURE); \
  }

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)1e-6 * tv.tv_usec;
}

char *get_source_code(const char *file_name, size_t *len) {
  char *source_code;
  size_t length;
  FILE *file = fopen(file_name, "r");
  if (file == NULL) {
    printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
    exit(EXIT_FAILURE);
  }

  fseek(file, 0, SEEK_END);
  length = (size_t)ftell(file);
  rewind(file);

  source_code = (char *)malloc(length + 1);
  fread(source_code, length, 1, file);
  source_code[length] = '\0';

  fclose(file);

  *len = length;
  return source_code;
}

void kmeans(int iteration_n, int class_n, int data_n, Point* centroids, Point* data, int* partitioned)
{

	cl_int err;	
	cl_platform_id platform;
	err = clGetPlatformIDs(1, &platform, NULL);
	CHECK_ERROR(err);

	cl_device_id device;
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	CHECK_ERROR(err);
	
	cl_context context;
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	CHECK_ERROR(err);
	
	cl_command_queue queue;
	queue = clCreateCommandQueue(context, device, 0, &err);
	CHECK_ERROR(err);

	cl_program program;
	size_t source_size;
	char *source_code = get_source_code("kernel.cl", &source_size);
	program = clCreateProgramWithSource(context, 1, (const char**)&source_code, &source_size, &err);
	CHECK_ERROR(err);
	
	err = clBuildProgram(program, 1, &device, "", NULL, NULL);
	if(err == CL_BUILD_PROGRAM_FAILURE){
		size_t log_size;
		char *log;
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		log = (char *)malloc(log_size + 1);
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		log[log_size] = '\0';
		printf("Compile error:\n%s\n", log);
		free(log);
	}
	CHECK_ERROR(err);
	
	cl_kernel kernel;
	kernel = clCreateKernel(program, "assign", &err);
	CHECK_ERROR(err);


	cl_mem buf_centroids, buf_data, buf_parts;
 	buf_centroids = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(Point) * class_n , NULL, &err);
	CHECK_ERROR(err);
 	buf_data = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(Point) * data_n  , NULL, &err);
	CHECK_ERROR(err);
 	buf_parts = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * data_n , NULL, &err);
	CHECK_ERROR(err);
	
	float dbl_max = DBL_MAX;
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_centroids);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_data);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_parts);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 3, sizeof(int), &class_n);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 4, sizeof(int), &data_n);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel, 5, sizeof(float), &dbl_max);
	CHECK_ERROR(err);

	size_t global_size = data_n;
	size_t local_size = 256;
	global_size = (global_size + local_size - 1) / local_size * local_size;
	//global_size[0] = (global_size[0] + local_size[0] -1) / local_size[0] * local_size[0];
	//global_size[1] = (global_size[1] + local_size[1] -1) / local_size[1] * local_size[1];
	
	double wr_start = get_time();
	err = clEnqueueWriteBuffer(queue, buf_centroids, CL_FALSE, 0, sizeof(Point) * class_n, centroids, 0,NULL, NULL);
	CHECK_ERROR(err);
	err = clEnqueueWriteBuffer(queue, buf_data, CL_FALSE, 0, sizeof(Point) * data_n, data, 0,NULL, NULL);
	CHECK_ERROR(err);
	clFinish(queue);
	double wr_end = get_time();
	
	int i;
    for (i = 0; i < iteration_n; i++) {
			double kernel_start = get_time();
			clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
			clFinish(queue);
			double kernel_end = get_time();

			clEnqueueReadBuffer(queue, buf_parts, CL_TRUE, 0, sizeof(int)*data_n, partitioned, 0, NULL,NULL);	

			////////////////////////////////////////////////////////////
			// Loop indices for iteration, data and class
			int  data_i, class_i;
			// Count number of data in each class
			int* count = (int*)malloc(sizeof(int) * class_n);
			// Update step
			// Clear sum buffer and class count
			for (class_i = 0; class_i < class_n; class_i++) {
					centroids[class_i].x = 0.0;
					centroids[class_i].y = 0.0;
					count[class_i] = 0;
			}
			// Sum up and count data for each class
			for (data_i = 0; data_i < data_n; data_i++) {         
					centroids[partitioned[data_i]].x += data[data_i].x;
					centroids[partitioned[data_i]].y += data[data_i].y;
					count[partitioned[data_i]]++;
			}

			printf("updata2 is OK \n");
			// Divide the sum with number of class for mean point
			for (class_i = 0; class_i < class_n; class_i++) {
					centroids[class_i].x /= count[class_i];
					centroids[class_i].y /= count[class_i];
			}
			err = clEnqueueWriteBuffer(queue, buf_centroids, CL_TRUE, 0, sizeof(Point) * class_n, centroids, 0,NULL, NULL);
			CHECK_ERROR(err);
	}
			clReleaseMemObject(buf_centroids);
			clReleaseMemObject(buf_data);
			clReleaseMemObject(buf_parts);
			clReleaseContext(context);
			clReleaseCommandQueue(queue);
			clReleaseProgram(program);
			clReleaseKernel(kernel);
}




