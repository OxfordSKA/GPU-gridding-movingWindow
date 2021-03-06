/*
 * Copyright (c) 2017, The University of Oxford
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "binary/oskar_binary.h"
#include "check_value.h"
#include "oskar_timer.h"
#include "oskar_grid_weights.h"
#include "oskar_grid_wproj.h"
#include "oskar_grid_wproj_gpu.hpp"
#include "generate_images.hpp"
#include "read_kernel.h"
#include "read_vis.h"
#include "write_fits_cube.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*===================================================================*/
/*===================================================================*/
/* Set this to 1 if calling a new version of the gridding function. */
#define HAVE_NEW_VERSION 1
#define WRITE_CUBE 0
#define WRITE_GRID 0
#define WRITE_DIFF_GRID 0
#define RUN_DIAGNOSTICS 0
#define WRITE_IMAGE_ORIG 0
#define WRITE_IMAGE_NEW 0
#define WRITE_IMAGE_DIFF 0
/*===================================================================*/
/*===================================================================*/

#define FLT sizeof(float)
#define DBL sizeof(double)
#define INT sizeof(int)

int main(int argc, char** argv)
{
    int i, status = 0;
    size_t j, num_cells = 0;
    size_t num_skipped = 0, num_skipped_orig = 0;
    double norm_orig = 0.0;
#if HAVE_NEW_VERSION
    size_t num_skipped_new = 0;
    double norm_new = 0.0;
#endif
    oskar_Timer *tmr_grid_vis_orig = 0, *tmr_grid_vis_new = 0;

    /* Visibility data. */
    oskar_Binary* h = 0;
    int coord_element_size, coord_precision;
    int vis_element_size, vis_precision, vis_type;
    int num_baselines, num_channels, num_pols, num_stations, num_times_total;
    int num_blocks, num_times_baselines, num_tags_per_block;
    int dim_start_and_size[6], max_times_per_block;
    double freq_start_hz;
    void *vis_block = 0, *uu = 0, *vv = 0, *ww = 0, *weight_1 = 0, *weight = 0;
    void *vis_grid_orig = 0, *vis_grid_new = 0, *weights_grid = 0;
    char *input_root = 0, *file_name = 0;

    // for diagnostics about grid/wsupport
    long long int *num_vis_per_grid, *num_vis_per_wsupport, *num_vis_per_wplane;
    float *avg_wsupport_per_grid;
    float *avg_wplane_per_grid;

    // for writing fits files
    int writeImageStatus=0;

    /* Kernel data. */
    void *kernels_real = 0, *kernels_imag = 0, *kernels = 0;
    int *support = 0, conv_size_half = 0, num_w_planes = 0, oversample = 0;
    int grid_size = 0;
    double w_scale = 0.0, cellsize_rad = 0.0;

    // data for testing double prec
    REAL *r_uu, *r_vv, *r_ww, *r_weight, *r_vis_block, *r_vis_grid_new, *r_vis_grid_orig, *r_kernels;

    /* Check that a test root name was given. */
    if (argc < 2)
    {
        fprintf(stderr, "Usage: ./main <test_data_root_name>\n");
        return EXIT_FAILURE;
    }

    /* Construct the visibility data file name. */
    input_root = argv[1];
    file_name = (char*) calloc(20 + strlen(input_root), 1);
    sprintf(file_name, "%s.vis", input_root);

    /* Open the visibility file for reading. */
    printf("Opening '%s'...\n", file_name);
    h = oskar_binary_create(file_name, 'r', &status);
    if (!h)
    {
        fprintf(stderr, "Failed to open visibility file '%s'\n", file_name);
        free(file_name);
        return EXIT_FAILURE;
    }

    /* Read relevant visibility header data. */
    oskar_binary_read_int(h, (unsigned char) OSKAR_TAG_GROUP_VIS_HEADER,
            OSKAR_VIS_HEADER_TAG_NUM_TAGS_PER_BLOCK, 0,
            &num_tags_per_block, &status);
    oskar_binary_read_int(h, (unsigned char) OSKAR_TAG_GROUP_VIS_HEADER,
            OSKAR_VIS_HEADER_TAG_AMP_TYPE, 0, &vis_type, &status);
    oskar_binary_read_int(h, (unsigned char) OSKAR_TAG_GROUP_VIS_HEADER,
            OSKAR_VIS_HEADER_TAG_COORD_PRECISION, 0, &coord_precision, &status);
    oskar_binary_read_int(h, (unsigned char) OSKAR_TAG_GROUP_VIS_HEADER,
            OSKAR_VIS_HEADER_TAG_MAX_TIMES_PER_BLOCK, 0,
            &max_times_per_block, &status);
    oskar_binary_read_int(h, (unsigned char) OSKAR_TAG_GROUP_VIS_HEADER,
            OSKAR_VIS_HEADER_TAG_NUM_TIMES_TOTAL, 0, &num_times_total, &status);
    oskar_binary_read_int(h, (unsigned char) OSKAR_TAG_GROUP_VIS_HEADER,
            OSKAR_VIS_HEADER_TAG_NUM_CHANNELS_TOTAL, 0, &num_channels, &status);
    oskar_binary_read_int(h, (unsigned char) OSKAR_TAG_GROUP_VIS_HEADER,
            OSKAR_VIS_HEADER_TAG_NUM_STATIONS, 0, &num_stations, &status);
    oskar_binary_read_double(h, (unsigned char) OSKAR_TAG_GROUP_VIS_HEADER,
            OSKAR_VIS_HEADER_TAG_FREQ_START_HZ, 0, &freq_start_hz, &status);

    /* Check for only one channel and one polarisation. */
    num_pols = oskar_type_is_scalar(vis_type) ? 1 : 4;
    if (num_channels != 1 || num_pols != 1)
    {
        fprintf(stderr, "These tests require single-channel, "
                "single-polarisation data.\n");
        oskar_binary_free(h);
        free(file_name);
        return EXIT_FAILURE;
    }

    /* Get data element sizes. */
    vis_precision      = oskar_type_precision(vis_type);
    vis_element_size   = (vis_precision == OSKAR_DOUBLE ? DBL : FLT);
    coord_element_size = (coord_precision == OSKAR_DOUBLE ? DBL : FLT);

    /* Calculate number of visibility blocks in the file. */
    num_blocks = (num_times_total + max_times_per_block - 1) /
            max_times_per_block;

    /* Arrays for visibility data and baseline coordinate arrays. */
    num_baselines = num_stations * (num_stations - 1) / 2;
    num_times_baselines = max_times_per_block * num_baselines;
    uu        = calloc(num_times_baselines, coord_element_size);
    vv        = calloc(num_times_baselines, coord_element_size);
    ww        = calloc(num_times_baselines, coord_element_size);
    weight    = calloc(num_times_baselines, vis_element_size);
    weight_1  = calloc(num_times_baselines, vis_element_size);
    vis_block = calloc(num_times_baselines, 2 * vis_element_size);

    printf("Num baselines: %d num stations: %d num times: %d\n", num_baselines, num_stations, max_times_per_block);
    /* Read convolution kernel data from FITS files. */
    sprintf(file_name, "%s_KERNELS_REAL.fits", input_root);
    kernels_real = read_kernel(file_name, vis_precision, &conv_size_half,
            &num_w_planes, &support, &oversample, &grid_size, &cellsize_rad,
            &w_scale, &status);
    sprintf(file_name, "%s_KERNELS_IMAG.fits", input_root);
    kernels_imag = read_kernel(file_name, vis_precision, &conv_size_half,
            &num_w_planes, NULL, &oversample, &grid_size, &cellsize_rad,
            &w_scale, &status);

    /* Convert kernels to complex values and generate unit weights. */
    num_cells = conv_size_half * conv_size_half * num_w_planes;
    kernels   = calloc(num_cells, 2 * vis_element_size);
    if (vis_precision == OSKAR_DOUBLE)
    {
        for (j = 0; j < num_cells; ++j)
        {
            ((double*)kernels)[2*j]     = ((const double*)kernels_real)[j];
            ((double*)kernels)[2*j + 1] = ((const double*)kernels_imag)[j];
        }
        for (i = 0; i < num_times_baselines; ++i)
            ((double*)weight_1)[i] = 1.0;
    }
    else
    {
        for (j = 0; j < num_cells; ++j)
        {
            ((float*)kernels)[2*j]     = ((const float*)kernels_real)[j];
            ((float*)kernels)[2*j + 1] = ((const float*)kernels_imag)[j];
        }
        for (i = 0; i < num_times_baselines; ++i)
            ((float*)weight_1)[i] = 1.0f;
    }

    /* Allocate weights grid and visibility grid. */
    num_cells = (size_t) grid_size;
    num_cells *= grid_size;
    weights_grid  = calloc(num_cells, coord_element_size);
    vis_grid_orig = calloc(num_cells, 2 * vis_element_size);
    vis_grid_new  = calloc(num_cells, 2 * vis_element_size);
    
    float* kernels_cube  = (float*) calloc(145*145*oversample*oversample*1201, 2 * vis_element_size);

    /* Create timers. */
    tmr_grid_vis_orig = oskar_timer_create(OSKAR_TIMER_NATIVE);
    tmr_grid_vis_new  = oskar_timer_create(OSKAR_TIMER_NATIVE);

    /* Loop over visibility blocks to generate the weights grid. */
    /* This is done for "uniform" visibility weighting. */
    if (!status) printf("Generating weights...\n");
    for (i = 0; i < num_blocks; ++i)
    {
        if (status)
        {
            fprintf(stderr, "Error (code %d)\n", status);
            break;
        }

        /* Read the visibility block metadata. */
        oskar_binary_set_query_search_start(h, i * num_tags_per_block, &status);
        oskar_binary_read(h, OSKAR_INT,
                (unsigned char) OSKAR_TAG_GROUP_VIS_BLOCK,
                OSKAR_VIS_BLOCK_TAG_DIM_START_AND_SIZE, i,
                INT*6, dim_start_and_size, &status);
        const int num_times = dim_start_and_size[2];
        const size_t block_size = num_times * num_baselines;

        /* Read and scale the baseline coordinates. */
        read_coords(h, i, coord_precision,
                block_size, uu, vv, ww, &status);
        scale_coords(freq_start_hz, coord_precision,
                block_size, uu, vv, ww);

        /* Update the weights grid, for "uniform" visibility weighting. */
        if (vis_precision == OSKAR_DOUBLE)
            oskar_grid_weights_write_d(block_size,
                    (const double*) uu, (const double*) vv,
                    (const double*) weight_1, cellsize_rad, grid_size,
                    &num_skipped, (double*) weights_grid);
        else
            oskar_grid_weights_write_f(block_size,
                    (const float*) uu, (const float*) vv,
                    (const float*) weight_1, (float) cellsize_rad, grid_size,
                    &num_skipped, (float*) weights_grid);
    }

    /* Loop over visibility blocks to generate the visibility grid. */
    if (!status) printf("Gridding visibilities...\n");
    for (i = 0; i < num_blocks; ++i)
    {
        if (status)
        {
            fprintf(stderr, "Error (code %d)\n", status);
            break;
        }

        /* Read the visibility block metadata. */
        oskar_binary_set_query_search_start(h, i * num_tags_per_block, &status);
        oskar_binary_read(h, OSKAR_INT,
                (unsigned char) OSKAR_TAG_GROUP_VIS_BLOCK,
                OSKAR_VIS_BLOCK_TAG_DIM_START_AND_SIZE, i,
                INT*6, dim_start_and_size, &status);
        const int num_times = dim_start_and_size[2];
        const size_t block_size = num_times * num_baselines;

        /* Read the visibility data. */
        oskar_binary_read(h, vis_type,
                (unsigned char) OSKAR_TAG_GROUP_VIS_BLOCK,
                OSKAR_VIS_BLOCK_TAG_CROSS_CORRELATIONS, i,
                coord_element_size*2 * block_size,
                vis_block, &status);

        /* Read and scale the baseline coordinates. */
        read_coords(h, i, coord_precision,
                block_size, uu, vv, ww, &status);
        for (int i=0; i<num_times_baselines; i++){
            //printf("ww: %f\n", ((float*) ww)[i]);
        }
        scale_coords(freq_start_hz, coord_precision,
                block_size, uu, vv, ww);

        /* Calculate new weights from weights grid. */
        /* This is done for "uniform" visibility weighting. */
        if (vis_precision == OSKAR_DOUBLE)
            oskar_grid_weights_read_d(block_size,
                    (const double*) uu, (const double*) vv,
                    (const double*) weight_1, (double*) weight,
                    cellsize_rad, grid_size, &num_skipped,
                    (const double*) weights_grid);
        else
            oskar_grid_weights_read_f(block_size,
                    (const float*) uu, (const float*) vv,
                    (const float*) weight_1, (float*) weight,
                    (float) cellsize_rad, grid_size, &num_skipped,
                    (const float*) weights_grid);

        /* Update new visibility grid. */
        oskar_timer_resume(tmr_grid_vis_new);
        /*===================================================================*/
        /*===================================================================*/
        /* CALL THE REPLACEMENT GRIDDING FUNCTION HERE. */

        printf("conv_size_half %d, wsupport max %d\n", conv_size_half, support[num_w_planes-1]);
 
        r_uu        = (REAL*) calloc(num_times_baselines, 2*coord_element_size);
        r_vv        = (REAL*) calloc(num_times_baselines, 2*coord_element_size);
        r_ww        = (REAL*) calloc(num_times_baselines, 2*coord_element_size);
        r_weight    = (REAL*) calloc(num_times_baselines, 2*vis_element_size);
        r_vis_block = (REAL*) calloc(num_times_baselines, 2*2 * vis_element_size);
        for (int i=0; i<num_times_baselines; i++){
            r_uu[i] = (REAL) ((float*)uu)[i];
            r_vv[i] = (REAL) ((float*)vv)[i];
            r_ww[i] = (REAL) ((float*)ww)[i];
            r_weight[i] = (REAL) ((float*)weight)[i];
            r_vis_block[2*i] = (REAL) ((float*)vis_block)[2*i];
            r_vis_block[2*i+1] = (REAL) ((float*)vis_block)[2*i+1];
        }

        num_cells = conv_size_half * conv_size_half * num_w_planes;
        r_kernels   = (REAL*) calloc(num_cells, 2*2 * vis_element_size);
        for (int i=0; i<num_cells*2; i++){
            r_kernels[i] = (REAL) ((float*)kernels)[i];
        }
           
        num_cells = grid_size*grid_size;
        r_vis_grid_new  = (REAL*) calloc(num_cells, 2*2 * vis_element_size);
        r_vis_grid_orig  = (REAL*) calloc(num_cells, 2*2 * vis_element_size);
        for (int i=0; i<num_cells*2; i++){
            r_vis_grid_new[i] = (REAL) ((float*)vis_grid_new)[i];
            r_vis_grid_orig[i] = (REAL) ((float*)vis_grid_orig)[i];
        }



#if HAVE_NEW_VERSION
        if (vis_precision == OSKAR_DOUBLE){
            /* Define a new name and call the new function. */
	    // double precision GPU version has not been implemented
            printf("RUNNING MOVING-WINDOW VERSION\n");
            oskar_grid_wproj_d(
                    (size_t) num_w_planes,
                    support,
                    oversample,
                    conv_size_half,
                    (const double*) kernels,
                    block_size,
                    (const double*) uu,
                    (const double*) vv,
                    (const double*) ww,
                    (const double*) vis_block,
                    (const double*) weight,
                    cellsize_rad,
                    w_scale,
                    grid_size,
                    &num_skipped_new,
                    &norm_new,
                    (double*) vis_grid_new);
        } else {
            printf("RUNNING MOVING-WINDOW VERSION\n");
            /* Define a new name and call the new function. */
            oskar_grid_wproj_gpu(
                    (size_t) num_w_planes,
                    support,
                    oversample,
                    conv_size_half,
                    (const REAL*) r_kernels,
                    block_size,
                    (const REAL*) r_uu,
                    (const REAL*) r_vv,
                    (const REAL*) r_ww,
                    (const REAL*) r_vis_block,
                    (const REAL*) r_weight,
                    (REAL) cellsize_rad,
                    (REAL) w_scale,
                    grid_size,
                    &num_skipped_new,
                    &norm_new,
                    (REAL*) r_vis_grid_new);
        }
#endif
        /*===================================================================*/
        /*===================================================================*/
        oskar_timer_pause(tmr_grid_vis_new);
#if 1

        /* Update the reference visibility grid. */
        oskar_timer_resume(tmr_grid_vis_orig);
        if (vis_precision == OSKAR_DOUBLE || 0){
            printf("RUNNING REFERENCE VERSION\n");
            oskar_grid_wproj_d(
                    (size_t) num_w_planes,
                    support,
                    oversample,
                    conv_size_half,
                    (const double*) r_kernels,
                    block_size,
                    (const double*) r_uu,
                    (const double*) r_vv,
                    (const double*) r_ww,
                    (const double*) r_vis_block,
                    (const double*) r_weight,
                    cellsize_rad,
                    w_scale,
                    grid_size,
                    &num_skipped_orig,
                    &norm_orig,
                    (double*) r_vis_grid_orig);
        } else {
            #if RUN_DIAGNOSTICS==1
                num_cells = grid_size*grid_size;
                num_vis_per_grid  = (long long int*) calloc(num_cells, sizeof(long long int));
                avg_wsupport_per_grid = (float*) calloc(num_cells, sizeof(float));
                avg_wplane_per_grid = (float*) calloc(num_cells, sizeof(float));
                int num_wsupport = support[num_w_planes-1]+1;
                num_vis_per_wsupport = (long long int*) calloc(num_wsupport, sizeof(long long int));
                num_vis_per_wplane = (long long int*) calloc(num_w_planes, sizeof(long long int));

                oskar_grid_wproj_f_diagnostics(
                        (size_t) num_w_planes,
                        support,
                        oversample,
                        conv_size_half,
                        (const float*) r_kernels,
                        block_size,
                        (const float*) r_uu,
                        (const float*) r_vv,
                        (const float*) r_ww,
                        (const float*) r_vis_block,
                        (const float*) r_weight,
                        (float) cellsize_rad,
                        (float) w_scale,
                        grid_size,
                        &num_skipped_orig,
                        &norm_orig,
                        (float*) r_vis_grid_orig,
                        num_vis_per_wsupport,
                        num_vis_per_wplane,
                        num_vis_per_grid,
                        avg_wsupport_per_grid,
                        avg_wplane_per_grid);
                for (int i=0; i<num_cells; i++){
                    if (num_vis_per_grid[i]!=0){
                        avg_wsupport_per_grid[i] = avg_wsupport_per_grid[i]/(float)num_vis_per_grid[i];
                        avg_wplane_per_grid[i] = avg_wplane_per_grid[i]/(float)num_vis_per_grid[i];
                    }
                }
                printf("writing avg wsupport\n");
                printf("WRITING WSUPPORT DISTRIBUTION\n");
                for (int i=0; i<num_wsupport; i++){
                    printf("%lld, %lld\n", i, num_vis_per_wsupport[i]);
                }
                printf("WRITING WPLANE DISTRIBUTION\n");
                for (int i=0; i<num_w_planes; i++){
                    printf("%lld, %lld\n", i, num_vis_per_wplane[i]);
                }
                write_fits_cube(OSKAR_SINGLE, avg_wsupport_per_grid, "avg_wsupport_per_grid", grid_size, grid_size, 1, 0, &writeImageStatus);
                write_fits_cube(OSKAR_SINGLE, avg_wplane_per_grid, "avg_wplane_per_grid", grid_size, grid_size, 1, 0, &writeImageStatus);
                    
            #else
            printf("RUNNING REFERENCE VERSION\n");
                oskar_grid_wproj_f(
                        (size_t) num_w_planes,
                        support,
                        oversample,
                        conv_size_half,
                        (const float*) r_kernels,
                        block_size,
                        (const float*) r_uu,
                        (const float*) r_vv,
                        (const float*) r_ww,
                        (const float*) r_vis_block,
                        (const float*) r_weight,
                        (float) cellsize_rad,
                        (float) w_scale,
                        grid_size,
                        &num_skipped_orig,
                        &norm_orig,
                        (float*) r_vis_grid_orig);
            #endif
            }
        oskar_timer_pause(tmr_grid_vis_orig);
#endif
    }

    /* Close the visibility data file. */
    oskar_binary_free(h);
    /* Compare visibility grids to check correctness. */
#if HAVE_NEW_VERSION
    /*
    if (!status)
    {
        printf("Checking grids...\n");
        if (vis_precision == OSKAR_DOUBLE)
        {
            const double* grid1 = (const double*)vis_grid_orig;
            const double* grid2 = (const double*)vis_grid_new;
            for (j = 0; j < num_cells; ++j)
            {
                if (!check_value_double(grid1[2*j], grid2[2*j]) ||
                        !check_value_double(grid1[2*j + 1], grid2[2*j + 1]))
                {
                    fprintf(stderr, "Inconsistent grid values (cell %lu).\n",
                            (unsigned long) j);
                    break;
                }
            }
        }
        else
        {
            const float* grid1 = (const float*)vis_grid_orig;
            const float* grid2 = (const float*)vis_grid_new;
            for (j = 0; j < num_cells; ++j)
            {
                if (!check_value_float(grid1[2*j], grid2[2*j]) ||
                        !check_value_float(grid1[2*j + 1], grid2[2*j + 1]))
                {
                    fprintf(stderr, "Inconsistent grid values (cell %lu).\n",
                            (unsigned long) j);
                    break;
                }
            }
        }
    }
    */
    #if WRITE_CUBE 
    printf("Writing kernels cube\n");
    //write_fits_cube(OSKAR_SINGLE|OSKAR_COMPLEX, kernels_cube, "kernels_cube", 145, 145, num_w_planes*oversample*oversample, 0, &writeImageStatus);
    write_fits_cube(OSKAR_SINGLE|OSKAR_COMPLEX, &((float*)kernels_cube)[0], "kernels_cube", 145*4, 145*4, 1, 0, &writeImageStatus);
    
    write_fits_cube(OSKAR_SINGLE|OSKAR_COMPLEX, &((float*)kernels)[2*2*conv_size_half*conv_size_half], "kernels_orig", conv_size_half, conv_size_half, 1, 0, &writeImageStatus);
#endif
 

#if WRITE_GRID
    printf("Writing new implementation image\n");
    write_fits_cube(OSKAR_SINGLE|OSKAR_COMPLEX, r_vis_grid_new, "vis_grid_new", grid_size, grid_size, 1, 0, &writeImageStatus);
    printf("Writing original implementation image\n");
    write_fits_cube(OSKAR_SINGLE|OSKAR_COMPLEX, r_vis_grid_orig, "vis_grid_orig", grid_size, grid_size, 1, 0, &writeImageStatus);
#endif

#if WRITE_IMAGE_ORIG
    printf("Generating orig image\n");
    float *image_orig = (float*) calloc(grid_size*grid_size, sizeof(float));
    generate_images_from_grid(grid_size, oversample, (float*) r_vis_grid_orig, image_orig, norm_orig);
    printf("Writing orig image\n");
    write_fits_cube(OSKAR_SINGLE, image_orig, "image_orig", grid_size, grid_size, 1, 0, &writeImageStatus);
    free(image_orig);
#endif

#if WRITE_IMAGE_NEW
    printf("Generating new image\n");
    float *image_new = (float*) calloc(grid_size*grid_size, sizeof(float));
    generate_images_from_grid(grid_size, oversample, (float*) r_vis_grid_new, image_new, norm_new);
    printf("Writing new image\n");
    write_fits_cube(OSKAR_SINGLE, image_new, "image_new", grid_size, grid_size, 1, 0, &writeImageStatus);
    free(image_new);
#endif

#if WRITE_IMAGE_DIFF
    printf("Generating new image\n");
    float *image_new_diff = (float*) calloc(grid_size*grid_size, sizeof(float));
    generate_images_from_grid(grid_size, oversample, (float*) r_vis_grid_new, image_new_diff, norm_new);

    printf("Generating orig image\n");
    float *image_orig_diff = (float*) calloc(grid_size*grid_size, sizeof(float));
    generate_images_from_grid(grid_size, oversample, (float*) r_vis_grid_orig, image_orig_diff, norm_orig);

    for (i=0; i<grid_size*grid_size; i++){
        image_orig_diff[i] = fabs(image_new_diff[i]-image_orig_diff[i]);
    }

    printf("Writing diff image\n");
    write_fits_cube(OSKAR_SINGLE, image_orig_diff, "image_diff", grid_size, grid_size, 1, 0, &writeImageStatus);
    free(image_orig_diff);
    free(image_new_diff);

#endif



#if HAVE_NEW_VERSION
    REAL* gridA = (REAL*)r_vis_grid_orig;
    REAL* gridB = (REAL*)r_vis_grid_new;
    REAL valAReal, valBReal, valAImag, valBImag;
    REAL maxDiffReal=0, diffReal, absDiffReal, RMSErrorReal=0;
    REAL maxDiffImag=0, diffImag, absDiffImag, RMSErrorImag=0;
    REAL maxRelDiffReal=0, relDiffReal, absRelDiffReal=0, RMSRelErrorReal=0;
    REAL maxRelDiffImag=0, relDiffImag, absRelDiffImag=0, RMSRelErrorImag=0;
    REAL normReal=0, normImag=0;
    for (j = 0; j < num_cells; ++j)
    {
        valAReal = gridA[2*j]; valAImag = gridA[2*j+1];
        valBReal = gridB[2*j]; valBImag = gridB[2*j+1];
        normReal += valAReal*valAReal;
        normImag += valAImag*valAImag;

        diffReal = valAReal - valBReal;
        diffImag = valAImag - valBImag;
        absDiffReal = fabs(diffReal);
        absDiffImag = fabs(diffImag);
        if (absDiffReal>maxDiffReal) maxDiffReal = absDiffReal;
        if (absDiffImag>maxDiffImag) maxDiffImag = absDiffImag;
        RMSErrorReal +=  diffReal*diffReal;
        RMSErrorImag +=  diffImag*diffImag;
        //gridA[2*j] = diffReal;
        //gridA[2*j+1] = diffImag;


        absRelDiffReal=0;
        if (fabs(valAReal)>fabs(valBReal)){
            relDiffReal = diffReal/valAReal;
            absRelDiffReal = fabs(relDiffReal);
        } else if (fabs(valBReal)>fabs(valAReal)){
            relDiffReal = diffReal/valBReal;
            absRelDiffReal = fabs(relDiffReal);
        }
        if (absRelDiffReal>maxRelDiffReal && absDiffReal>10e-13) maxRelDiffReal = absRelDiffReal;
         
        absRelDiffImag=0;
        if (fabs(valAImag)>fabs(valBImag)){
            relDiffImag = diffImag/valAImag;
            absRelDiffImag = fabs(relDiffImag);
        } else if (fabs(valBImag)>fabs(valAImag)){
            relDiffImag = diffImag/valBImag;
            absRelDiffImag = fabs(relDiffImag);
        }
        if (absRelDiffImag>maxRelDiffImag && absDiffImag>10e-13) maxRelDiffImag = absRelDiffImag;

        gridA[2*j] = absRelDiffReal;
        gridA[2*j+1] = absRelDiffImag;
    }

    //RMSErrorReal = sqrtf(RMSErrorReal)/sqrtf(normReal);
    //RMSErrorImag = sqrtf(RMSErrorImag)/sqrtf(normImag);
    REAL final_error = sqrtf(RMSErrorReal+RMSErrorImag)/sqrtf(normReal+normImag);
    printf("\nError: %.14f\n\n", final_error);
#if WRITE_DIFF_GRID
    /*
    printf("REAL: RMSError %f, Max Error %f Max Rel. Error %f\n", RMSErrorReal, maxDiffReal, maxRelDiffReal);
    printf("IMAG: RMSError %f, Max Error %f Max Rel. Error %f\n", RMSErrorImag, maxDiffImag, maxRelDiffImag);
    */
    printf("Writing diff (original-new) implementation image\n");
    write_fits_cube(OSKAR_SINGLE|OSKAR_COMPLEX, gridA, "vis_grid_diff", grid_size, grid_size, 1, 0, &writeImageStatus);
    //write_fits_cube(OSKAR_DOUBLE|OSKAR_COMPLEX, gridA, "vis_grid_diff", grid_size, grid_size, 1, 0, &writeImageStatus);
#endif
#endif
 
#else
    printf("No new version of visibility grid to check.\n");
#endif


    /* Free memory. */
    free(support);
    free(kernels_real);
    free(kernels_imag);
    free(kernels);
    free(file_name);
    free(uu);
    free(vv);
    free(ww);
    free(weight);
    free(weight_1);
    free(vis_block);
    free(vis_grid_orig);
    free(vis_grid_new);
    free(weights_grid);

    free(r_uu);
    free(r_vv);
    free(r_ww);
    free(r_weight);
    free(r_vis_block);
    free(r_vis_grid_new);
    free(r_vis_grid_orig);
    free(r_kernels);

#if WRITE_CUBE
    free(kernels_cube);
#endif

    /* Report timing. */
    printf("Gridding visibilities took %.3f seconds (original version)\n",
            oskar_timer_elapsed(tmr_grid_vis_orig));
#if HAVE_NEW_VERSION
    printf("Gridding visibilities took %.3f seconds (new version)\n",
            oskar_timer_elapsed(tmr_grid_vis_new));
#endif
    oskar_timer_free(tmr_grid_vis_orig);
    oskar_timer_free(tmr_grid_vis_new);
    return 0;
}
