typedef struct Point {
    float x, y;
}Point;


__kernel void assign (
				 __global Point *centroids, 
				 __global Point *data,
				 __global int* partitioned,
				 __const int class_n,
				 __const int data_n,
				 __const float dbl_max)
{
	int data_i = get_global_id(0);
	Point t;
	float min_dist = dbl_max; 	
	
	int class_i;
	for(class_i = 0; class_i < class_n; class_i++){
			 t.x = data[data_i].x - centroids[class_i].x;
			 t.y = data[data_i].y - centroids[class_i].y;

			 float dist = t.x * t.x + t.y * t.y;
		
			 if (dist < min_dist) {
			 	partitioned[data_i] = class_i;
				min_dist = dist;
			 }
	}
}

