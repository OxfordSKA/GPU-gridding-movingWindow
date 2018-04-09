#include "oskar_grid_functions_spheroidal.h"
#include "oskar_grid_correction.h"
#include "oskar_fftphase.h"

#include <cufft.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void generate_images_from_grid(int grid_size, int oversample,
        float *grid, float *image, double plane_norm)
{
    size_t num_cells;
    num_cells = grid_size * grid_size;

    printf("grid: %f\n", grid[1000]);

    double *grid_d = (double*) calloc(2*num_cells, sizeof(double));

    /* Apply normalisation. */
    if (plane_norm > 0.0 || plane_norm < 0.0){
        for (int i=0; i<num_cells; i++){
            grid_d[2*i] =  ((double) grid[2*i])/plane_norm;
        }
    }
    printf("plane_norm: %f\n", plane_norm);

    /* Perform FFT shift of the input grid. */
    oskar_fftphase_cd(grid_size, grid_size, grid_d);

    /* Copy mem to GPU */
    double *gpu_grid_d;
    cudaMalloc((void**)&gpu_grid_d, 2*num_cells*sizeof(double));
    cudaMemcpy(gpu_grid_d, grid_d, 2*num_cells*sizeof(double), cudaMemcpyHostToDevice);

    /* Call FFT on GPU */
    cufftHandle cufft_plan;
    cufftPlan2d(&cufft_plan, grid_size, grid_size, CUFFT_Z2Z);
    cufftExecZ2Z(cufft_plan, (cufftDoubleComplex*)gpu_grid_d, (cufftDoubleComplex*)gpu_grid_d, CUFFT_FORWARD);

    /* Copy mem back from GPU */
    cudaMemcpy(grid_d, gpu_grid_d, 2*num_cells*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(gpu_grid_d);
    
    /* Generate grid correction function if required. */
    //! does this need to be 2d (num_cells?)
    double *corr_func = (double*) calloc(grid_size, sizeof(double));
    oskar_grid_correction_function_spheroidal(grid_size, oversample, corr_func);

    printf("corr_func: %f\n", corr_func[100]);

    /* FFT shift again, and apply grid correction. */
    oskar_fftphase_cd(grid_size, grid_size, grid_d);
    oskar_grid_correction_d(grid_size, corr_func, grid_d);
       
    /* copy result to image array in single prec */ 
    for (int i=0; i<num_cells; i++){
        image[i] = (float) grid_d[2*i];
    }

    printf("image: %f\n", image[1000]);
    free(grid_d);
}

